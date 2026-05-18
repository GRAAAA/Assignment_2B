import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

DATA_FILE   = os.path.join(os.path.dirname(__file__), "Scats_Data_October_2006.xls")
SHEET_NAME  = "Data"
TEST_SIZE   = 0.2
RANDOM_SEED = 42
SEQ_LEN     = 12  # 12 x 15 min = 3-hour look-back window for LSTM/GRU

DROP_COLS = [
    "SCATS Number", "CD_MELWAY", "HF VicRoads Internal",
    "VR Internal Stat", "VR Internal Loc", "NB_TYPE_SURVEY",
]

TIME_COLS = [f"V{i:02d}" for i in range(96)]

# Cache parsed Excel to avoid re-reading the ~8 MB file repeatedly
_RAW_CACHE = {}


def _read_excel_raw(filepath: str = DATA_FILE) -> pd.DataFrame:
    if filepath not in _RAW_CACHE:
        _RAW_CACHE[filepath] = pd.read_excel(
            filepath, sheet_name=SHEET_NAME, header=1, engine="xlrd")
    return _RAW_CACHE[filepath]


def load_raw(filepath: str = DATA_FILE) -> pd.DataFrame:
    df = _read_excel_raw(filepath).copy()
    drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_site_coords(filepath: str = DATA_FILE) -> dict:
    """Return {scats_id: (lat, lon)} for all sites in the dataset."""
    df = _read_excel_raw(filepath)
    coords = {}
    for _, row in df.iterrows():
        sid = str(row.get("SCATS Number", "")).strip()
        lat = row.get("NB_LATITUDE", 0)
        lon = row.get("NB_LONGITUDE", 0)
        if sid and sid.lower() != "nan" and pd.notna(lat) and lat != 0:
            coords[sid] = (float(lat), float(lon))
    return coords


def get_locations(filepath: str = DATA_FILE) -> np.ndarray:
    return load_raw(filepath)["Location"].unique()


def reshape_to_timeseries(df: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Convert one SCATS location from wide format (96 V-columns per day)
    to a long-format time series with one row per 15-min slot.
    """
    df_loc     = df[df["Location"] == location]
    value_cols = [c for c in TIME_COLS if c in df_loc.columns]

    out = df_loc.melt(id_vars=["Date"], value_vars=value_cols,
                      var_name="slot_col", value_name="flow")
    out["time_slot"] = out["slot_col"].str[1:].astype(int)
    # Normalise date to midnight so slot N maps exactly to N*15 min past midnight
    out["datetime"] = (out["Date"].dt.normalize()
                       + pd.to_timedelta(out["time_slot"] * 15, unit="m"))
    out["flow"]     = out["flow"].fillna(0.0).astype(float)

    dt = out["datetime"].dt
    out["hour"]        = dt.hour
    out["minute"]      = dt.minute
    out["day_of_week"] = dt.dayofweek
    out["day"]         = dt.day
    out["month"]       = dt.month
    out["year"]        = dt.year

    cols = ["datetime", "flow", "time_slot", "hour", "minute",
            "day_of_week", "day", "month", "year"]
    return out[cols].sort_values("datetime").reset_index(drop=True)


def build_features(ts: pd.DataFrame, lag1: bool = True, lag2: bool = True) -> pd.DataFrame:
    df = ts.copy()
    if lag1:
        df["y_lag1"] = df["flow"].shift(1)
    if lag2:
        df["y_lag2"] = df["flow"].shift(2)
    df.dropna(inplace=True)
    return df


def get_rf_features_target(ts: pd.DataFrame):
    feat = build_features(ts)
    feature_cols = ["day", "month", "year", "day_of_week",
                    "time_slot", "hour", "y_lag1", "y_lag2"]
    X = feat[feature_cols].values
    y = feat["flow"].values
    return X, y, feature_cols


def get_rf_train_test(ts: pd.DataFrame):
    X, y, cols = get_rf_features_target(ts)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    return X_tr, X_te, y_tr, y_te, cols


def build_sequences(values: np.ndarray, seq_len: int = SEQ_LEN):
    """Slide a window of length seq_len over a 1-D series to build (X, y) pairs."""
    X_list, y_list = [], []
    for i in range(len(values) - seq_len):
        X_list.append(values[i: i + seq_len])
        y_list.append(values[i + seq_len])
    return np.array(X_list).reshape(-1, seq_len, 1), np.array(y_list)


def get_dl_train_test(ts: pd.DataFrame, seq_len: int = SEQ_LEN):
    flow   = ts["flow"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(flow).flatten()
    X, y   = build_sequences(scaled, seq_len)
    split  = int(len(X) * (1 - TEST_SIZE))
    return X[:split], X[split:], y[:split], y[split:], scaler


def prepare_location(loc_index: int, filepath: str = DATA_FILE):
    df  = load_raw(filepath)
    loc = df["Location"].unique()
    if loc_index >= len(loc):
        raise IndexError(f"loc_index {loc_index} out of range (max {len(loc)-1})")
    location = loc[loc_index]
    ts = reshape_to_timeseries(df, location)
    return {
        "location": location,
        "ts":       ts,
        "rf_data":  get_rf_train_test(ts),
        "dl_data":  get_dl_train_test(ts),
    }


if __name__ == "__main__":
    print("Loading data...")
    data = prepare_location(1)
    print(f"  Location : {data['location']}")
    ts = data["ts"]
    print(f"  Rows     : {len(ts)}")
    print(f"  Flow range: {ts['flow'].min():.0f} – {ts['flow'].max():.0f}")
    X_tr, X_te, y_tr, y_te, cols = data["rf_data"]
    print(f"  RF train : {X_tr.shape}  test: {X_te.shape}")
    X_tr_dl, X_te_dl, _, _, _ = data["dl_data"]
    print(f"  DL train : {X_tr_dl.shape}  test: {X_te_dl.shape}")
