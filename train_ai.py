#!/usr/bin/env python3
"""
train_ai.py – LSTM forecast using 1-minute downsampled data
• Reads raw 5-second sensor data from DB
• Resamples to 1-minute means
• Dynamically picks the largest look-back window (up to 7 days)
• Trains or loads LSTM to predict 24 hours (1440 steps)
• Downsamples forecast to 1-hour resolution (24 values)
• Saves 24 hourly predictions to SensorPrediction
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

# ---------------- Django setup ----------------
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData, SensorPrediction  # noqa: E402

# ---------------- Config ----------------
TARGET_FREQ          = "1min"
MAX_LOOKBACK_STEPS   = 7 * 24 * 60         # max 7 days = 10,080 steps
FORECAST_STEPS       = 24 * 60             # predict 1440 min (24 h)
MIN_SEQS_FOR_LSTM    = 10
HISTORY_DAYS         = 7
MODEL_PATH           = "models/lstm_sensor_1min.h5"



SENSOR_COLS = [
    "temperature",
    "humidity",
    "oxygen_level",
    "co2_level",
    "light_illumination",
]

# ---------------- Load raw sensor data ----------------
now = datetime.utcnow()
qs = (
    SensorData.objects
    .filter(timestamp__gte=now - timedelta(days=HISTORY_DAYS))
    .order_by("timestamp")
)

row_count = qs.count()
print(f"DEBUG • fetched {row_count} raw SensorData rows")

if row_count < 2:
    raise RuntimeError("Not enough SensorData rows to train.")

df = pd.DataFrame.from_records(qs.values("timestamp", *SENSOR_COLS))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# ---------------- Resample to 1-minute bins ----------------
df = df.resample(TARGET_FREQ).mean().dropna()
print("DEBUG • after 1-min resample:", len(df), "rows")

available_rows = len(df)
if available_rows < MIN_SEQS_FOR_LSTM + 1:
    raise RuntimeError(f"Need at least {MIN_SEQS_FOR_LSTM + 1} rows, got {available_rows}.")
# ---------------- Dynamic look-back (adjusted for actual learning) ----------------
MIN_SEQS_FOR_LSTM = 100
lookback_steps = max(10, available_rows - MIN_SEQS_FOR_LSTM)

# Cap at max 1,440 (1 day), which is enough for 100+ samples
lookback_steps = min(lookback_steps, 1440)

print(
    f"DEBUG • adjusted look-back = {lookback_steps} steps "
    f"({lookback_steps/60:.1f} h), "
    f"training sequences = {available_rows - lookback_steps}"
)

# ---------------- Dynamic look-back ----------------
lookback_steps = min(
    MAX_LOOKBACK_STEPS,
    available_rows - MIN_SEQS_FOR_LSTM
)
print(
    f"DEBUG • look-back = {lookback_steps} steps "
    f"({lookback_steps/60:.1f} h), "
    f"training sequences = {available_rows - lookback_steps}"
)

# ---------------- Scale ----------------
data = df[SENSOR_COLS].to_numpy(dtype=np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ---------------- Create sequences ----------------
def build_sequences(arr: np.ndarray, steps: int):
    if len(arr) <= steps:
        return (
            np.empty((0, steps, arr.shape[1]), dtype=np.float32),
            np.empty((0, arr.shape[1]), dtype=np.float32)
        )
    X = np.stack([arr[i:i+steps] for i in range(len(arr) - steps)], axis=0).astype("float32")
    Y = arr[steps:].astype("float32")
    return X, Y

X, Y = build_sequences(scaled, lookback_steps)
print("DEBUG • X shape:", X.shape, "Y shape:", Y.shape)

# ---------------- Train or baseline ----------------
if len(X) < MIN_SEQS_FOR_LSTM:
    print(f"⚠️  Only {len(X)} training samples – using persistence baseline.")
    last_obs = df.iloc[-1][SENSOR_COLS].to_numpy(dtype=np.float32)
    predictions = np.tile(last_obs, (FORECAST_STEPS, 1))
else:
    # Train/val split
    split = int(0.9 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val,   Y_val   = X[split:], Y[split:]

    # Build or load model
    if os.path.exists(MODEL_PATH) and lookback_steps == MAX_LOOKBACK_STEPS:
        model = keras.models.load_model(MODEL_PATH)
        print("DEBUG • loaded existing model")
    else:
        model = Sequential([
            Input(shape=(lookback_steps, len(SENSOR_COLS))),
            LSTM(64, activation="tanh"),
            Dense(len(SENSOR_COLS))
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(clipnorm=1.0),
            loss="mse"
        )

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=12,
        batch_size=4,
        verbose=1
    )

    if lookback_steps == MAX_LOOKBACK_STEPS:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)

    # Forecast 1440 1-minute steps
    seq = scaled[-lookback_steps:].copy()
    pred_scaled = []
    for _ in range(FORECAST_STEPS):
        p = model.predict(seq[np.newaxis], verbose=0)[0]
        pred_scaled.append(p)
        seq = np.concatenate([seq[1:], p[None, :]], axis=0)

    pred_scaled = np.asarray(pred_scaled, dtype=np.float32)
    predictions = scaler.inverse_transform(pred_scaled)
    predictions += data[-1] - predictions[0]  # continuity shift

# ---------------- Clamp temperature ----------------
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)

# ---------------- Downsample to hourly + write to DB ----------------
last_ts = df.index[-1]
selected_predictions = predictions[59::60][:24]  # 1 prediction/hour

timestamps = [last_ts + timedelta(hours=i + 1) for i in range(len(selected_predictions))]

records = [
    SensorPrediction(
        timestamp=ts,
        temperature=round(float(r[0]), 2),
        humidity=round(float(r[1]), 2),
        oxygen_level=round(float(r[2]), 2),
        co2_level=round(float(r[3]), 2),
        light_illumination=round(float(r[4]), 2),
    )
    for ts, r in zip(timestamps, selected_predictions)
]

with transaction.atomic():
    SensorPrediction.objects.all().delete()
    SensorPrediction.objects.bulk_create(records, batch_size=24)

print(f"✅ Wrote {len(records)} hourly prediction rows to SensorPrediction")
