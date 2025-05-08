#!/usr/bin/env python3
"""
robust_forecast.py – 24‑hour sensor forecast via Django ORM
• Screams in DEBUG if the database returns 0 rows
• Shrinks look‑back automatically; falls back to persistence baseline
• Trains / reloads an LSTM when you finally have enough sequences
• Saves predictions atomically to SensorPrediction
"""

import os
import django
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from django.db import transaction
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow import keras

# ---------------------------------------------------------------------
# DJANGO SETUP
# ---------------------------------------------------------------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData, SensorPrediction  # noqa: E402

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
HISTORY_DAYS        = 7      # look back this many days in the DB
MAX_LOOKBACK        = 24     # preferred hour window for LSTM
MIN_SEQS_FOR_LSTM   = 10     # need ≥ this many training samples for LSTM
MODEL_PATH          = "models/lstm_sensor.h5"

SENSOR_COLS = [
    "temperature",
    "humidity",
    "oxygen_level",
    "co2_level",
    "light_illumination",
]

# ---------------------------------------------------------------------
# OPTIONAL: turn on raw SQL logging (remove if too noisy)
# ---------------------------------------------------------------------
if os.getenv("SQL_DEBUG", "0") == "1":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("django.db.backends").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------
# 1. READ FROM DATABASE
# ---------------------------------------------------------------------
now = datetime.utcnow()
qs = (
    SensorData.objects
    .filter(timestamp__gte=now - timedelta(days=HISTORY_DAYS))
    .order_by("timestamp")
)

row_count = qs.count()
print(f"DEBUG • fetched {row_count} SensorData rows")

if row_count == 0:
    raise RuntimeError("Database returned zero rows – aborting forecast.")

# show two sample rows so you can *see* values
for rec in qs[:2]:
    print(
        "DEBUG • sample row:",
        rec.timestamp.isoformat(),
        rec.temperature,
        rec.humidity,
        rec.oxygen_level,
        rec.co2_level,
        rec.light_illumination,
    )

# build DataFrame
df = pd.DataFrame.from_records(qs.values("timestamp", *SENSOR_COLS))

# ---------------------------------------------------------------------
# 2. RESAMPLE TO HOURLY MEANS
# ---------------------------------------------------------------------
df_hourly = (
    df.set_index("timestamp")
      .sort_index()
      .resample("1H", origin="start")
      .mean()
      .dropna()                # drop any row that still has NaNs
)

available_rows = len(df_hourly)
print(f"DEBUG • hourly rows after resample: {available_rows}")

if available_rows < 2:
    raise RuntimeError(
        f"Need ≥ 2 hourly rows to forecast, only have {available_rows}."
    )

# ---------------------------------------------------------------------
# 3. DYNAMIC LOOK‑BACK WINDOW
# ---------------------------------------------------------------------
LOOKBACK = max(1, min(MAX_LOOKBACK, available_rows - 1))
print(f"DEBUG • using LOOKBACK={LOOKBACK} hours")

data = df_hourly[SENSOR_COLS].to_numpy(dtype=np.float32)

# ---------------------------------------------------------------------
# 4. SCALE FEATURES 0‑1
# ---------------------------------------------------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ---------------------------------------------------------------------
# 5. MAKE (X, Y) SEQUENCES
# ---------------------------------------------------------------------
def make_xy(arr: np.ndarray, lkbk: int):
    """
    Build training pairs with strict shape checks.
    X  -> (n_samples, lkbk, n_features)
    Y  -> (n_samples, n_features)
    """
    if len(arr) <= lkbk:
        return (
            np.empty((0, lkbk, arr.shape[1]), dtype=np.float32),
            np.empty((0, arr.shape[1]), dtype=np.float32),
        )

    X = np.stack(
        [arr[i : i + lkbk] for i in range(len(arr) - lkbk)],
        axis=0,
    ).astype("float32")
    Y = arr[lkbk:].astype("float32")
    return X, Y


X, Y = make_xy(scaled, LOOKBACK)
print("DEBUG • X shape:", X.shape, "Y shape:", Y.shape)

# ---------------------------------------------------------------------
# 6. CHOOSE MODEL STRATEGY
# ---------------------------------------------------------------------
if len(X) < MIN_SEQS_FOR_LSTM:
    # ——— Persistence baseline ———
    print(
        f"⚠️  Only {len(X)} training samples (<{MIN_SEQS_FOR_LSTM}). "
        "Repeating last observation for 24 h."
    )
    last_obs = df_hourly.iloc[-1][SENSOR_COLS].to_numpy(dtype=np.float32)
    predictions = np.tile(last_obs, (24, 1))
else:
    # ---------------- Train / val split ----------------
    if len(X) >= 5:
        split = int(0.8 * len(X))
        X_train, Y_train = X[:split], Y[:split]
        X_val,   Y_val   = X[split:], Y[split:]
    else:
        X_train, Y_train = X, Y
        X_val,   Y_val   = None, None

    # ---------------- Build / load model ----------------
    if (
        os.path.exists(MODEL_PATH)
        and LOOKBACK == MAX_LOOKBACK
    ):
        model = keras.models.load_model(MODEL_PATH)
        print("DEBUG • loaded existing model from disk")
    else:
        model = Sequential(
            [
                Input(shape=(LOOKBACK, len(SENSOR_COLS))),
                LSTM(64, activation="relu"),
                Dense(len(SENSOR_COLS)),
            ]
        )
        model.compile(optimizer="adam", loss="mse")

    # ---------------- Train ----------------
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val) if X_val is not None else None,
        epochs=30,
        batch_size=min(8, len(X_train)),
        verbose=1,
    )

    # save only if shape is stable
    if LOOKBACK == MAX_LOOKBACK:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        print("DEBUG • model saved to", MODEL_PATH)

    # ---------------- Forecast next 24 h ----------------
    seq = scaled[-LOOKBACK:].copy()
    pred_scaled = []
    for _ in range(24):
        p = model.predict(seq[np.newaxis], verbose=0)[0]
        pred_scaled.append(p)
        seq = np.concatenate([seq[1:], p[None, :]], axis=0)

    predictions = scaler.inverse_transform(np.asarray(pred_scaled))

    # shift so first forecast matches last real obs
    predictions += df_hourly.iloc[-1].to_numpy(dtype=np.float32) - predictions[0]

# ---------------------------------------------------------------------
# 7. POST‑PROCESSING
# ---------------------------------------------------------------------
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)  # temp safety clamp

# ---------------------------------------------------------------------
# 8. WRITE PREDICTIONS TO DB
# ---------------------------------------------------------------------
last_ts = df_hourly.index[-1]
records = [
    SensorPrediction(
        timestamp=last_ts + timedelta(hours=i + 1),
        temperature=round(float(r[0]), 2),
        humidity=round(float(r[1]), 2),
        oxygen_level=round(float(r[2]), 2),
        co2_level=round(float(r[3]), 2),
        light_illumination=round(float(r[4]), 2),
    )
    for i, r in enumerate(predictions)
]

with transaction.atomic():
    SensorPrediction.objects.all().delete()
    SensorPrediction.objects.bulk_create(records, batch_size=24)

print("✅ 24‑hour forecast saved: wrote", len(records), "rows to SensorPrediction")
