#!/usr/bin/env python3
"""
train_ai.py  –  1-minute-resolution 24-h forecast with dynamic look-back

* Down-samples raw 5-second sensor data to 1-minute means
* Look-back window = min(available_rows – 1, 7 days)  (max 10 080 steps)
* Forecast horizon fixed at 1 440 steps (next 24 h @ 1 min)
* LSTM trains when ≥ 10 training sequences; otherwise uses persistence baseline
* Writes 1 440 prediction rows into SensorPrediction atomically
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
TARGET_FREQ          = "1min"         # one-minute bins
MAX_LOOKBACK_STEPS   = 7 * 24 * 60    # 10 080  (7 days @ 1 min)
FORECAST_STEPS       = 24 * 60        # 1 440   (24 h  @ 1 min)
MIN_SEQS_FOR_LSTM    = 10             # need ≥ 10 training samples
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
# 1. READ RAW DATA (5-SECOND CADENCE)
# ---------------------------------------------------------------------
now = datetime.utcnow()
qs = (
    SensorData.objects
    .filter(timestamp__gte=now - timedelta(days=HISTORY_DAYS))
    .order_by("timestamp")
)

row_count = qs.count()
print(f"DEBUG • fetched {row_count} raw SensorData rows")

if row_count < 2:
    raise RuntimeError("Database returned < 2 rows – cannot forecast.")

df = pd.DataFrame.from_records(qs.values("timestamp", *SENSOR_COLS))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# ---------------------------------------------------------------------
# 2. DOWNSAMPLE TO 1-MINUTE BINS
# ---------------------------------------------------------------------
df = df.resample(TARGET_FREQ).mean().dropna()
print("DEBUG • after 1-min resample:", len(df), "rows")

available_rows = len(df)
if available_rows < 2:
    raise RuntimeError("After resampling, not enough data to continue.")

# ---------------------------------------------------------------------
# 3. DYNAMIC LOOK-BACK WINDOW  ← REPLACE THIS WHOLE SECTION
# ---------------------------------------------------------------------
MIN_SEQS_FOR_LSTM = 10                # keep it configurable

if available_rows <= MIN_SEQS_FOR_LSTM:
    raise RuntimeError(
        f"Need > {MIN_SEQS_FOR_LSTM} rows to train, have {available_rows}."
    )

lookback_steps = min(
    MAX_LOOKBACK_STEPS,
    available_rows - MIN_SEQS_FOR_LSTM  # leave enough rows for training
)

print(
    f"DEBUG • look-back = {lookback_steps} steps "
    f"({lookback_steps/60:.1f} h), "
    f"training sequences = {available_rows - lookback_steps}"
)

# ---------------------------------------------------------------------
# 4. SCALE FEATURES
# ---------------------------------------------------------------------
data = df[SENSOR_COLS].to_numpy(dtype=np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ---------------------------------------------------------------------
# 5. BUILD (X, Y) SEQUENCES
# ---------------------------------------------------------------------
def build_sequences(arr: np.ndarray, steps: int):
    """Return X (n, steps, f) and Y (n, f)."""
    if len(arr) <= steps:
        return np.empty((0, steps, arr.shape[1]), dtype=np.float32), np.empty(
            (0, arr.shape[1]), dtype=np.float32
        )
    X = np.stack(
        [arr[i : i + steps] for i in range(len(arr) - steps)], axis=0
    ).astype("float32")
    Y = arr[steps:].astype("float32")
    return X, Y


X, Y = build_sequences(scaled, lookback_steps)
print("DEBUG • X shape:", X.shape, "Y shape:", Y.shape)

# ---------------------------------------------------------------------
# 6. CHOOSE STRATEGY
# ---------------------------------------------------------------------
if len(X) < MIN_SEQS_FOR_LSTM:
    # — Persistence baseline —
    print(
        f"⚠️  Only {len(X)} training samples (<{MIN_SEQS_FOR_LSTM}). "
        "Repeating last observed values for the next 24 h."
    )
    last_obs   = df.iloc[-1][SENSOR_COLS].to_numpy(dtype=np.float32)
    predictions = np.tile(last_obs, (FORECAST_STEPS, 1))
else:
    # -------- Train / load LSTM --------
    split = int(0.9 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val,   Y_val   = X[split:], Y[split:]

    if os.path.exists(MODEL_PATH) and lookback_steps == MAX_LOOKBACK_STEPS:
        model = keras.models.load_model(MODEL_PATH)
        print("DEBUG • loaded existing model")
    else:
        model = Sequential([
            Input(shape=(lookback_steps, len(SENSOR_COLS))),
            LSTM(64, activation="relu"),
            Dense(len(SENSOR_COLS)),
        ])
        model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=12,
        batch_size=4,
        verbose=1,
    )

    if lookback_steps == MAX_LOOKBACK_STEPS:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)

    # -------- Forecast --------
    seq = scaled[-lookback_steps:].copy()
    pred_scaled = []
    for _ in range(FORECAST_STEPS):
        p = model.predict(seq[np.newaxis], verbose=0)[0]
        pred_scaled.append(p)
        seq = np.concatenate([seq[1:], p[None, :]], axis=0)

    pred_scaled = np.asarray(pred_scaled, dtype=np.float32)
    predictions = scaler.inverse_transform(pred_scaled)

    # align first forecast with last real obs
    predictions += data[-1] - predictions[0]

# ---------------------------------------------------------------------
# 7. POST-PROCESSING
# ---------------------------------------------------------------------
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)  # clamp temperature

# ---------------------------------------------------------------------
# 8. SAVE 1 440 ONE-MINUTE PREDICTIONS
# ---------------------------------------------------------------------
last_ts = df.index[-1]
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

print("✅ Wrote", len(records), "1-min prediction rows to SensorPrediction")
