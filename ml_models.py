from model_rf   import train_rf,   predict_rf, get_feature_importances
try:
    from model_lstm import build_lstm, train_lstm, predict_lstm
    from model_gru  import build_gru,  train_gru,  predict_gru
except ImportError:
    build_lstm = train_lstm = predict_lstm = None
    build_gru  = train_gru  = predict_gru  = None

__all__ = [
    "build_lstm", "train_lstm", "predict_lstm",
    "build_gru",  "train_gru",  "predict_gru",
    "train_rf",   "predict_rf", "get_feature_importances",
    "compare_models",
]


def compare_models(results: list) -> dict:
    """Pick the best model by lowest NRMSE. results is a list of train_* return dicts."""
    summary = sorted([r["metrics"] for r in results], key=lambda m: m["nrmse"])
    return {
        "summary":    summary,
        "best_model": summary[0]["model"],
        "best_nrmse": summary[0]["nrmse"],
        "best_rmse":  summary[0]["rmse"],
        "best_mae":   summary[0].get("mae", None),
        "best_r2":    summary[0].get("r2",  None),
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_processor import prepare_location

    print("Preparing data for location 1 ...")
    data = prepare_location(1)

    print("Training Random Forest ...")
    rf_res = train_rf(data["rf_data"])
    print(f"  RF   RMSE={rf_res['metrics']['rmse']:.4f}  NRMSE={rf_res['metrics']['nrmse']:.4f}")

    print("Training LSTM (10 epochs) ...")
    lstm_res = train_lstm(data["dl_data"], epochs=10, verbose=0)
    print(f"  LSTM RMSE={lstm_res['metrics']['rmse']:.4f}  NRMSE={lstm_res['metrics']['nrmse']:.4f}")

    print("Training GRU (10 epochs) ...")
    gru_res = train_gru(data["dl_data"], epochs=10, verbose=0)
    print(f"  GRU  RMSE={gru_res['metrics']['rmse']:.4f}  NRMSE={gru_res['metrics']['nrmse']:.4f}")

    cmp = compare_models([lstm_res, gru_res, rf_res])
    print(f"\nBest model : {cmp['best_model']}")
    print(f"Best NRMSE : {cmp['best_nrmse']:.4f}")
    print("\nFull ranking:")
    for i, s in enumerate(cmp["summary"], 1):
        print(f"  {i}. {s['model']:<15}  RMSE={s['rmse']:.4f}  NRMSE={s['nrmse']:.4f}  MAE={s.get('mae',0):.4f}  R²={s.get('r2',0):.4f}")
    print("ml_models OK")
