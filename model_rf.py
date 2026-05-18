import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_processor import prepare_location, get_rf_train_test


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


def train_rf(rf_data: tuple, n_estimators: int = 200,
             max_depth: int = 15, random_state: int = 42) -> dict:
    X_tr, X_te, y_tr, y_te, feature_cols = rf_data

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    importances = dict(sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True))

    return {
        "model":        model,
        "feature_cols": feature_cols,
        "importances":  importances,
        "metrics":      _compute_metrics(y_te, y_pred, "Random Forest"),
        "y_test":       y_te,
        "y_pred":       y_pred,
    }


def predict_rf(model, feature_cols: list,
               day: int, month: int, year: int,
               day_of_week: int, time_slot: int, hour: int,
               y_lag1: float = 0.0, y_lag2: float = 0.0) -> float:
    x    = np.array([[day, month, year, day_of_week,
                      time_slot, hour, y_lag1, y_lag2]])
    pred = float(model.predict(x)[0])
    return max(0.0, pred)


def get_feature_importances(result: dict) -> None:
    print("\nFeature Importances (Random Forest):")
    print(f"  {'Feature':<15} {'Importance':>10}")
    for feat, imp in result["importances"].items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<15} {imp:>10.4f}  {bar}")


if __name__ == "__main__":
    data   = prepare_location(1)
    result = train_rf(data["rf_data"])
    m      = result["metrics"]
    print(f"RF    RMSE={m['rmse']:.4f}  NRMSE={m['nrmse']:.4f}  "
          f"MAE={m['mae']:.4f}  R2={m['r2']:.4f}")
    get_feature_importances(result)
    pred = predict_rf(result["model"], result["feature_cols"],
                      day=1, month=10, year=2006,
                      day_of_week=6, time_slot=32, hour=8,
                      y_lag1=45.0, y_lag2=42.0)
    print(f"Sample prediction (Oct 1 2006, 08:00): {pred:.1f} vehicles")
