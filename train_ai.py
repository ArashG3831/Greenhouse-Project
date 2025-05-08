#!/usr/bin/env python3
"""
train_ai.py  – 24-hour forecast at 1-minute resolution
• Pulls raw 5-second rows, resamples to 1-minute means
• Uses a 1-day look-back (1 440 mins) ⇒ 1-day forecast (1 440 rows)
• Falls back to persistence baseline only if the DB has < 2 × look-back rows
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
RESAMPLE_FREQ        = "1T"              # 1-minute buckets
LOOKBACK_STEPS       = 24 * 60           # 1 440 minutes = 24 h
FORECAST_STEPS       = LOOKBACK_STEPS    # predict next 24 h
MIN_SEQS_FOR_LSTM    = 10                # train once we have ≥ 10 samples
HISTORY_DAYS         = 7
MODEL_PATH           = "models/lstm_sensor_1min.h5"

SENSOR_COLS = [
    "temperature",
    "humidity",
    "oxygen_level",
    "co2_level",
    "light_illumination",
]

# ---------------------------------------------------------------------
# 1. READ RAW DATA (5-second cadence)
# ---------------------------------------------------------------------
now = datetime.utcnow()
qs = (
    SensorData.objects
    .filter(timestamp__gte=now - timedelta(days=HISTORY_DAYS))
    .order_by("timestamp")
)

row_count = qs.count()
print(f"DEBUG • fetched {row_count} raw SensorData rows")

if row_count < LOOKBACK_STEPS * 2:
    raise RuntimeError(
        f"Need ≥ {LOOKBACK_STEPS*2} rows (~48 h) to forecast, "
        f"only have {row_count}.")
)

df = pd.DataFrame.from_records(qs.values("timestamp", *SENSOR_COLS))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# ---------------------------------------------------------------------
# 2. RESAMPLE TO 1-MINUTE MEANS (forward-fill gaps)
# ---------------------------------------------------------------------
df_1min = (
    df.resample(RESAMPLE_FREQ, origin="start")
      .mean()
      .ffill()
      .dropna()
)

print("DEBUG • rows after 1-min resample:", len(df_1min))

data = df_1min[SENSOR_COLS].to_numpy(dtype=np.float32)

# ---------------------------------------------------------------------
# 3. SCALE FEATURES 0-1
# ---------------------------------------------------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ---------------------------------------------------------------------
# 4. MAKE (X, Y) SEQUENCES
# ---------------------------------------------------------------------
def make_xy(arr: np.ndarray, steps: int):
    if len(arr) <= steps:
        return (
            np.empty((0, steps, arr.shape[1]), dtype=np.float32),
            np.empty((0, arr.shape[1]), dtype=np.float32),
        )
    X = np.stack([arr[i : i + steps] for i in range(len(arr) - steps)], axis=0)
    Y = arr[steps:]
    return X.astype("float32"), Y.astype("float32")


X, Y = make_xy(scaled, LOOKBACK_STEPS)
print("DEBUG • X shape:", X.shape, "Y shape:", Y.shape)

# ---------------------------------------------------------------------
# 5. CHOOSE MODEL STRATEGY
# ---------------------------------------------------------------------
if len(X) < MIN_SEQS_FOR_LSTM:
    # —— Persistence baseline ——
    print(
        f"⚠️  Only {len(X)} training samples (<{MIN_SEQS_FOR_LSTM}). "
        "Repeating last observation for 24 h."
    )
    last_obs = df_1min.iloc[-1][SENSOR_COLS].to_numpy(dtype=np.float32)
    predictions = np.tile(last_obs, (FORECAST_STEPS, 1))
else:
    # ---- Split train / val (90 % / 10 %) ----
    split = int(0.9 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val,   Y_val   = X[split:], Y[split:]

    # ---- Build / load model ----
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print("DEBUG • loaded existing model")
    else:
        model = Sequential([
            Input(shape=(LOOKBACK_STEPS, len(SENSOR_COLS))),
            LSTM(64, activation="relu"),
            Dense(len(SENSOR_COLS)),
        ])
        model.compile(optimizer="adam", loss="mse")

    # ---- Train ----
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=15,               # bump epochs if you have GPU time
        batch_size=4,            # 4 × 1 440 × 5 ≈ 0.03 GB per batch
        verbose=1,
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    # ---- Forecast ----
    seq = scaled[-LOOKBACK_STEPS:].copy()
    pred_scaled = []
    for _ in range(FORECAST_STEPS):
        p = model.predict(seq[np.newaxis], verbose=0)[0]
        pred_scaled.append(p)
        seq = np.concatenate([seq[1:], p[None, :]], axis=0)

    pred_scaled = np.asarray(pred_scaled, dtype=np.float32)
    predictions = scaler.inverse_transform(pred_scaled)
    predictions += data[-1] - predictions[0]  # level shift

# ---------------------------------------------------------------------
# 6. POST-PROCESS
# ---------------------------------------------------------------------
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)

# ---------------------------------------------------------------------
# 7. SAVE 1 440 PREDICTION ROWS
# ---------------------------------------------------------------------
last_ts = df_1min.index[-1]
timestamps = [
    last_ts + timedelta(minutes=i + 1) for i in range(FORECAST_STEPS)
]

records = [
    SensorPrediction(
        timestamp=ts,
        temperature=round(float(r[0]), 2),
        humidity=round(float(r[1]), 2),
        oxygen_level=round(float(r[2]), 2),
        co2_level=round(float(r[3]), 2),
        light_illumination=round(float(r[4]), 2),
    )
    for ts, r in zip(timestamps, predictions)
]

with transaction.atomic():
    SensorPrediction.objects.all().delete()
    SensorPrediction.objects.bulk_create(records, batch_size=256)

print("✅ wrote", len(records), "1-min prediction rows to SensorPrediction")
