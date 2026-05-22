import argparse
import gc
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(__file__), ".matplotlib-cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(os.path.dirname(__file__), ".cache"),
)

import data_processor as dp


def _json_print(payload):
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _history_for_prediction(ts, predict_day, time_slot):
    today = datetime.strptime(predict_day, "%m/%d/%Y")
    dow = today.weekday()
    seq_len = dp.SEQ_LEN
    ts_sorted = ts.sort_values("datetime").reset_index(drop=True)
    match = ts_sorted[
        (ts_sorted["day_of_week"] == dow)
        & (ts_sorted["time_slot"] == time_slot)
    ]
    if len(match) > 0:
        idx = match.index[0]
        window = ts_sorted["flow"].values[max(0, idx - seq_len):idx]
        if len(window) < seq_len:
            pad = np.full(seq_len - len(window),
                          ts_sorted["flow"].values[:seq_len].mean())
            window = np.concatenate([pad, window])
        return np.array(window[-seq_len:], dtype=float)
    return np.array(ts_sorted["flow"].values[-seq_len:], dtype=float)


def train(args):
    os.makedirs(args.artifact_dir, exist_ok=True)
    data = dp.prepare_location(args.loc_index, args.data_file)

    # Suppress noisy Keras progress so stdout remains machine-readable JSON.
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        results = {}
        if args.model in ("both", "lstm"):
            from model_lstm import train_lstm
            lstm_res = train_lstm(data["dl_data"], epochs=args.epochs, verbose=0)
            lstm_path = os.path.join(args.artifact_dir, "lstm.keras")
            lstm_res["model"].save(lstm_path)
            results["lstm"] = {
                "model_path": lstm_path,
                "metrics": lstm_res["metrics"],
            }
            tf.keras.backend.clear_session()
            del lstm_res
            gc.collect()

        if args.model in ("both", "gru"):
            from model_gru import train_gru
            gru_res = train_gru(data["dl_data"], epochs=args.epochs, verbose=0)
            gru_path = os.path.join(args.artifact_dir, "gru.keras")
            gru_res["model"].save(gru_path)
            results["gru"] = {
                "model_path": gru_path,
                "metrics": gru_res["metrics"],
            }
            tf.keras.backend.clear_session()
            del gru_res
            gc.collect()

    payload = {
        "ok": True,
        "location": data["location"],
    }
    payload.update(results)
    _json_print(payload)


def predict(args):
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        from tensorflow.keras.models import load_model
        from model_lstm import predict_lstm
        from model_gru import predict_gru

        df = dp.load_raw(args.data_file)
        ts = dp.reshape_to_timeseries(df, args.location_name)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(ts["flow"].values.reshape(-1, 1))
        history = _history_for_prediction(ts, args.predict_day, args.time_slot)

        model = load_model(args.model_path)
        if args.model_type == "lstm":
            flow = predict_lstm(model, scaler, history, dp.SEQ_LEN)
        else:
            flow = predict_gru(model, scaler, history, dp.SEQ_LEN)

    _json_print({"ok": True, "flow": float(flow)})


def main():
    parser = argparse.ArgumentParser(
        description="Run TensorFlow deep-model work outside the Tk GUI process.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-file", default=dp.DATA_FILE)
    p_train.add_argument("--loc-index", type=int, default=1)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--artifact-dir", required=True)
    p_train.add_argument("--model", choices=["both", "lstm", "gru"],
                         default="both")

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--data-file", default=dp.DATA_FILE)
    p_predict.add_argument("--model-type", choices=["lstm", "gru"], required=True)
    p_predict.add_argument("--model-path", required=True)
    p_predict.add_argument("--location-name", required=True)
    p_predict.add_argument("--predict-day", required=True)
    p_predict.add_argument("--time-slot", type=int, required=True)

    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args)
        else:
            predict(args)
    except Exception as exc:
        _json_print({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
