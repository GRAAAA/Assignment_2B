import os
import sys
import json
import signal
import subprocess
import warnings
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import data_processor as dp
from model_rf import train_rf, predict_rf

train_lstm = predict_lstm = train_gru = predict_gru = None


def _deep_worker_python() -> str:
    """Prefer the project-local clean TensorFlow env when it has been created."""
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, ".venv-tf", "bin", "python"),
        os.path.join(base_dir, ".venv-tf", "Scripts", "python.exe"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return sys.executable


def compare_models(results: list) -> dict:
    """Pick the best model by lowest NRMSE."""
    summary = sorted([r["metrics"] for r in results], key=lambda m: m["nrmse"])
    return {
        "summary":    summary,
        "best_model": summary[0]["model"],
        "best_nrmse": summary[0]["nrmse"],
        "best_rmse":  summary[0]["rmse"],
        "best_mae":   summary[0].get("mae", None),
        "best_r2":    summary[0].get("r2", None),
    }


class TrafficPredictor:
    """
    Trains LSTM, GRU and Random Forest for traffic flow prediction.
    Call train_all() once, then predict() for any date/time/location.
    """

    def __init__(self, data_file: str = None):
        self.data_file   = data_file or dp.DATA_FILE
        self.artifact_dir = os.path.join(os.path.dirname(__file__), "model_artifacts")
        self.loc_index   = None
        self.location    = None
        self.ts          = None
        self.lstm_result = None
        self.gru_result  = None
        self.rf_result   = None
        self.comparison  = None
        self._trained    = False
        self._ts_cache   = {}   # cache time series per location name
        self._station_rf = {}   # per-station RF models
        self._external_deep_learning = False
        self._dl_pred_cache = {}

    def _run_deep_worker(self, args: list) -> dict:
        cmd = [_deep_worker_python(), os.path.join(os.path.dirname(__file__),
                                                   "deep_model_worker.py")] + args
        env = os.environ.copy()
        env.update({
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        })
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        payload = None
        for line in reversed(lines):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "ok" in candidate:
                payload = candidate
                break
        if proc.returncode != 0 or not payload or not payload.get("ok"):
            if payload:
                detail = payload.get("error")
            else:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                if proc.returncode < 0:
                    sig_num = -proc.returncode
                    sig_name = signal.Signals(sig_num).name
                    exit_info = f"terminated by {sig_name} ({sig_num})"
                else:
                    exit_info = f"exit code {proc.returncode}"
                detail = "; ".join(
                    part for part in [
                        exit_info,
                        f"stderr: {stderr}" if stderr else "",
                        f"stdout: {stdout}" if stdout else "",
                        f"command: {' '.join(cmd)}",
                    ] if part
                )
            raise RuntimeError(f"Deep-model worker failed: {detail}")
        return payload

    def _get_location_ts(self, location_name: str):
        if location_name not in self._ts_cache:
            df = dp.load_raw(self.data_file)
            self._ts_cache[location_name] = dp.reshape_to_timeseries(df, location_name)
        return self._ts_cache[location_name]

    def _get_station_rf(self, location_name: str):
        if location_name not in self._station_rf:
            ts = self._get_location_ts(location_name)
            self._station_rf[location_name] = train_rf(dp.get_rf_train_test(ts))
        return self._station_rf[location_name]

    def train_station_rf_models(self, location_names: list,
                                progress=None) -> dict:
        total = len(location_names)
        for i, name in enumerate(location_names, 1):
            if name not in self._station_rf:
                self._get_station_rf(name)
            if progress is not None:
                progress(i, total, name)
        return self._station_rf

    def train_all(self, loc_index: int = 1,
                  epochs: int = 50, verbose: int = 0,
                  include_deep_learning: bool = True,
                  strict_deep_learning: bool = True,
                  external_deep_learning: bool = False) -> dict:
        print(f"[TrafficPredictor] Loading location index {loc_index}...")
        data = dp.prepare_location(loc_index, self.data_file)
        self.loc_index = loc_index
        self.location  = data["location"]
        self.ts        = data["ts"]
        print(f"  Location: {self.location}  ({len(self.ts)} records)")

        print("[TrafficPredictor] Training Random Forest...")
        self.rf_result = train_rf(data["rf_data"])
        m = self.rf_result["metrics"]
        print(f"  RF   RMSE={m['rmse']:.2f}  NRMSE={m['nrmse']:.3f}")

        if not include_deep_learning:
            print("[TrafficPredictor] LSTM/GRU skipped (RF-only mode).")
            self.lstm_result = None
            self.gru_result = None
        else:
            try:
                self._external_deep_learning = external_deep_learning
                if external_deep_learning:
                    print("[TrafficPredictor] Training LSTM/GRU in TensorFlow worker...")
                    lstm_payload = self._run_deep_worker([
                        "train",
                        "--data-file", self.data_file,
                        "--loc-index", str(loc_index),
                        "--epochs", str(epochs),
                        "--artifact-dir", self.artifact_dir,
                        "--model", "lstm",
                    ])
                    gru_payload = self._run_deep_worker([
                        "train",
                        "--data-file", self.data_file,
                        "--loc-index", str(loc_index),
                        "--epochs", str(epochs),
                        "--artifact-dir", self.artifact_dir,
                        "--model", "gru",
                    ])
                    self.lstm_result = {
                        "external": True,
                        "model_path": lstm_payload["lstm"]["model_path"],
                        "metrics": lstm_payload["lstm"]["metrics"],
                    }
                    self.gru_result = {
                        "external": True,
                        "model_path": gru_payload["gru"]["model_path"],
                        "metrics": gru_payload["gru"]["metrics"],
                    }
                    for label, result in (("LSTM", self.lstm_result),
                                          ("GRU", self.gru_result)):
                        m = result["metrics"]
                        print(f"  {label:<4} RMSE={m['rmse']:.2f}  NRMSE={m['nrmse']:.3f}")
                else:
                    global train_lstm, predict_lstm, train_gru, predict_gru
                    if train_lstm is None or train_gru is None:
                        from model_lstm import train_lstm as _train_lstm, predict_lstm as _predict_lstm
                        from model_gru import train_gru as _train_gru, predict_gru as _predict_gru
                        train_lstm = _train_lstm
                        predict_lstm = _predict_lstm
                        train_gru = _train_gru
                        predict_gru = _predict_gru

                    print("[TrafficPredictor] Training LSTM...")
                    self.lstm_result = train_lstm(data["dl_data"], epochs=epochs,
                                                  verbose=verbose)
                    m = self.lstm_result["metrics"]
                    print(f"  LSTM RMSE={m['rmse']:.2f}  NRMSE={m['nrmse']:.3f}")

                    print("[TrafficPredictor] Training GRU...")
                    self.gru_result = train_gru(data["dl_data"], epochs=epochs,
                                                verbose=verbose)
                    m = self.gru_result["metrics"]
                    print(f"  GRU  RMSE={m['rmse']:.2f}  NRMSE={m['nrmse']:.3f}")
            except Exception as e:
                if strict_deep_learning:
                    raise RuntimeError(
                        "LSTM/GRU training failed in the TensorFlow worker. "
                        f"Details: {e}"
                    ) from e
                print(f"  [WARN] LSTM/GRU skipped ({type(e).__name__}): RF-only mode.")
                self.lstm_result = None
                self.gru_result  = None

        results = [r for r in (self.lstm_result, self.gru_result,
                               self.rf_result) if r is not None]
        self.comparison = compare_models(results)
        print(f"[TrafficPredictor] Best model: {self.comparison['best_model']} "
              f"(NRMSE={self.comparison['best_nrmse']:.3f})")
        self._trained = True
        return self.comparison

    def predict(self, predict_day: str, time_slot: int,
                model: str = "best", location_name: str = None) -> float:
        if not self._trained:
            raise RuntimeError("Call train_all() before predict().")

        model = model.lower()
        if model == "best":
            model = self.comparison["best_model"].lower().replace(" ", "_")
            if "random" in model:
                model = "rf"
        if model not in ("lstm", "gru", "rf"):
            raise ValueError("model must be one of: best, lstm, gru, rf")

        ts = (self.ts if location_name is None
              else self._get_location_ts(location_name))

        if model in ("lstm", "gru"):
            return self._predict_dl(predict_day, time_slot, model, ts,
                                    location_name=location_name)
        return self._predict_rf(predict_day, time_slot, ts,
                                location_name=location_name)

    def _predict_rf(self, predict_day: str, time_slot: int, ts=None,
                    location_name: str = None) -> float:
        today = datetime.strptime(predict_day, "%m/%d/%Y")
        hour  = (time_slot * 15) // 60
        dow   = today.weekday()

        if location_name is not None:
            station      = self._get_station_rf(location_name)
            rf_model     = station["model"]
            feature_cols = station["feature_cols"]
        else:
            rf_model     = self.rf_result["model"]
            feature_cols = self.rf_result["feature_cols"]

        if ts is None:
            ts = self.ts

        # Estimate lag features from historical mean for same day-of-week
        slot_lag1 = (time_slot - 1) % 96
        m1 = (ts["day_of_week"] == dow) & (ts["time_slot"] == slot_lag1)
        if not m1.any():
            m1 = ts["time_slot"] == slot_lag1
        y_lag1 = float(ts.loc[m1, "flow"].mean()) if m1.any() else 0.0

        slot_lag2 = (time_slot - 2) % 96
        m2 = (ts["day_of_week"] == dow) & (ts["time_slot"] == slot_lag2)
        if not m2.any():
            m2 = ts["time_slot"] == slot_lag2
        y_lag2 = float(ts.loc[m2, "flow"].mean()) if m2.any() else 0.0

        return predict_rf(rf_model, feature_cols,
                          day=today.day, month=today.month, year=today.year,
                          day_of_week=dow, time_slot=time_slot, hour=hour,
                          y_lag1=y_lag1, y_lag2=y_lag2)

    def _predict_dl(self, predict_day: str, time_slot: int,
                    which: str, ts=None, location_name: str = None) -> float:
        today   = datetime.strptime(predict_day, "%m/%d/%Y")
        dow     = today.weekday()
        seq_len = dp.SEQ_LEN
        if ts is None:
            ts = self.ts

        result = self.lstm_result if which == "lstm" else self.gru_result
        if result is None:
            raise RuntimeError(f"{which.upper()} model not available. Use model='rf'.")
        if result.get("external"):
            worker_location = location_name or self.location
            cache_key = (which, worker_location, predict_day, int(time_slot),
                         result["model_path"])
            if cache_key in self._dl_pred_cache:
                return self._dl_pred_cache[cache_key]
            payload = self._run_deep_worker([
                "predict",
                "--data-file", self.data_file,
                "--model-type", which,
                "--model-path", result["model_path"],
                "--location-name", worker_location,
                "--predict-day", predict_day,
                "--time-slot", str(time_slot),
            ])
            flow = float(payload["flow"])
            self._dl_pred_cache[cache_key] = flow
            return flow

        ts_sorted = ts.sort_values("datetime").reset_index(drop=True)
        match     = ts_sorted[(ts_sorted["day_of_week"] == dow) &
                               (ts_sorted["time_slot"] == time_slot)]

        if len(match) > 0:
            idx    = match.index[0]
            window = ts_sorted["flow"].values[max(0, idx - seq_len): idx]
            if len(window) < seq_len:
                pad    = np.full(seq_len - len(window),
                                 ts_sorted["flow"].values[:seq_len].mean())
                window = np.concatenate([pad, window])
            history = window[-seq_len:]
        else:
            history = ts_sorted["flow"].values[-seq_len:]

        history = np.array(history, dtype=float)

        # Use this location's own scaler so the predicted magnitude fits its range
        loc_scaler = MinMaxScaler(feature_range=(0, 1))
        loc_scaler.fit(ts["flow"].values.reshape(-1, 1))

        if which == "lstm":
            return predict_lstm(self.lstm_result["model"], loc_scaler,
                                history, seq_len)
        return predict_gru(self.gru_result["model"], loc_scaler,
                           history, seq_len)

    def predict_for_location(self, loc_index: int, predict_day: str,
                             time_slot: int, model: str = "best") -> float:
        if loc_index != self.loc_index:
            self.train_all(loc_index)
        return self.predict(predict_day, time_slot, model)


_default_predictor = None


def predict_flow(predict_day: str, loc_index: int, time_slot: int,
                 model: str = "best") -> float:
    """Module-level convenience: predict_flow("10/1/2006", 1, 32)."""
    global _default_predictor
    if _default_predictor is None or _default_predictor.loc_index != loc_index:
        _default_predictor = TrafficPredictor()
        _default_predictor.train_all(loc_index)
    return _default_predictor.predict(predict_day, time_slot, model)


if __name__ == "__main__":
    tp = TrafficPredictor()
    tp.train_all(loc_index=1, epochs=15)
    print("\nPrediction tests:")
    for slot in [0, 16, 32, 64]:
        h = (slot * 15) // 60
        m = (slot * 15) % 60
        print(f"  {h:02d}:{m:02d}  RF={tp.predict('10/1/2006', slot, 'rf'):5.1f}"
              f"  LSTM={tp.predict('10/1/2006', slot, 'lstm'):5.1f}"
              f"  GRU={tp.predict('10/1/2006', slot, 'gru'):5.1f}")
