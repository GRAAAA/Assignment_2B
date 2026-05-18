import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    raise ImportError("TensorFlow required: pip install tensorflow")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_processor import SEQ_LEN, get_dl_train_test, prepare_location


def _compute_metrics(y_true, y_pred, name):
    rmse      = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_true = float(np.mean(y_true))
    return {
        "model": name,
        "rmse":  round(rmse, 4),
        "nrmse": round(rmse / mean_true if mean_true != 0 else float("inf"), 4),
        "mae":   round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2":    round(float(r2_score(y_true, y_pred)), 4),
    }


def build_lstm(seq_len: int = SEQ_LEN, units: int = 64) -> Sequential:
    model = Sequential([
        LSTM(units, input_shape=(seq_len, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(units // 2),
        Dropout(0.2),
        Dense(1),
    ], name="LSTM_traffic")
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(dl_data: tuple, epochs: int = 50,
               batch_size: int = 32, verbose: int = 0) -> dict:
    X_tr, X_te, y_tr, y_te, scaler = dl_data

    model = build_lstm(seq_len=X_tr.shape[1])
    early_stop = EarlyStopping(monitor="val_loss", patience=8,
                               restore_best_weights=True)
    history = model.fit(X_tr, y_tr, validation_split=0.1,
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stop], verbose=verbose)

    y_pred_scaled = model.predict(X_te, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

    return {
        "model":   model,
        "scaler":  scaler,
        "history": history,
        "metrics": _compute_metrics(y_true, y_pred, "LSTM"),
        "y_test":  y_true,
        "y_pred":  y_pred,
    }


def predict_lstm(model, scaler, flow_history: np.ndarray,
                 seq_len: int = SEQ_LEN) -> float:
    recent        = np.array(flow_history[-seq_len:], dtype=float).reshape(-1, 1)
    recent_scaled = scaler.transform(recent).flatten()
    X             = recent_scaled.reshape(1, seq_len, 1)
    pred_scaled   = model.predict(X, verbose=0).flatten()[0]
    pred          = scaler.inverse_transform([[pred_scaled]])[0][0]
    train_max     = float(scaler.data_max_[0])
    train_min     = float(scaler.data_min_[0])
    return max(0.0, float(min(max(pred, train_min), train_max * 1.1)))


if __name__ == "__main__":
    data   = prepare_location(1)
    result = train_lstm(data["dl_data"], epochs=10, verbose=1)
    m      = result["metrics"]
    print(f"LSTM  RMSE={m['rmse']:.4f}  NRMSE={m['nrmse']:.4f}  "
          f"MAE={m['mae']:.4f}  R2={m['r2']:.4f}")
    ts   = data["ts"]
    pred = predict_lstm(result["model"], result["scaler"],
                        ts["flow"].values[:SEQ_LEN])
    print(f"Sample prediction: {pred:.1f} vehicles")
