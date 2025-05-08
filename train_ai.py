#!/usr/bin/env python3
"""
robust_forecast.py  –  24-hour sensor forecast via Django ORM
* Handles as little as two hourly rows (persistence baseline)
* Trains/loads an LSTM once you have ≥ MIN_SEQS_FOR_LSTM sequences
* Saves predictions back to SensorPrediction in one atomic transaction
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

from sensor.models import SensorData, SensorPrediction   # noqa: E402

# ---------------- Config ----------------
HISTORY_DAYS        = 7      # query this much history from DB
MAX_LOOKBACK        = 24     # desired timesteps (hours) for the LSTM
MIN_SEQS_FOR_LSTM   = 10     # min training samples to bother with an LSTM
MODEL_PATH          = "models/lstm_sensor.h5"

SENSOR_COLS = [
    "temperature",
    "humidity",
    "oxygen_level",
    "co2_level",
    "light_illumination",
]

# ---------------- Load & resample ----------------
now = datetime.utcnow()
qs = (
    SensorData.objects
    .filter(timestamp__gte=now - timedelta(days=HISTORY_DAYS))
    .order_by("timestamp")
)
df = pd.DataFrame.from_records(qs.values("timestamp", *SENSOR_COLS))
if df.empty:
    raise RuntimeError("No SensorData rows found in the last 7 days.")

df_hourly = (
    df.set_index("timestamp")
      .sort_index()
      .resample("1H", origin="start")
      .mean()
      .dropna()
)

available_rows = len(df_hourly)
if available_rows < 2:
    raise RuntimeError(
        f"Need ≥ 2 hourly rows to forecast, only have {available_rows}."
    )

# ---------------- Dynamic look-back ----------------
LOOKBACK = max(1, min(MAX_LOOKBACK, available_rows - 1))

data = df_hourly[SENSOR_COLS].to_numpy(dtype=np.float32)

# ---------------- Scale ----------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ---------------- Build sequences ----------------
def make_xy(arr, lkbk):
    X, Y = [], []
    for i in range(len(arr) - lkbk):
        X.append(arr[i : i + lkbk])
        Y.append(arr[i + lkbk])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)

X, Y = make_xy(scaled, LOOKBACK)

# ---------------- Decide modelling strategy ----------------
if len(X) < MIN_SEQS_FOR_LSTM:
    # ----- Persistence baseline -----
    print(
        f"⚠️  Only {len(X)} training samples – using persistence baseline."
    )
    last_obs   = df_hourly.iloc[-1][SENSOR_COLS].to_numpy(dtype=np.float32)
    predictions = np.tile(last_obs, (24, 1))
else:
    # ----- Prepare train / val split -----
    split = int(len(X) * 0.8) if len(X) >= 5 else 0
    X_train, Y_train = (X[:split], Y[:split]) if split else (X, Y)
    X_val,   Y_val   = (X[split:], Y[split:]) if split else (None, None)

    print("X_train", X_train.shape, "Y_train", Y_train.shape)

    # ----- Build / load model -----
    if (
        os.path.exists(MODEL_PATH)
        and LOOKBACK == MAX_LOOKBACK
        and len(X) >= MIN_SEQS_FOR_LSTM
    ):
        model = keras.models.load_model(MODEL_PATH)
        print("Loaded existing model.")
    else:
        model = Sequential(
            [
                Input(shape=(LOOKBACK, len(SENSOR_COLS))),
                LSTM(64, activation="relu"),
                Dense(len(SENSOR_COLS)),
            ]
        )
        model.compile(optimizer="adam", loss="mse")

    # ----- Train -----
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val) if X_val is not None else None,
        epochs=30,
        batch_size=min(8, len(X_train)),
        verbose=1,
    )

    # Save the model only when full look-back is used (consistent shape)
    if LOOKBACK == MAX_LOOKBACK:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)

    # ----- Forecast -----
    seq = scaled[-LOOKBACK:].copy()
    pred_scaled = []
    for _ in range(24):
        p = model.predict(seq[np.newaxis], verbose=0)[0]
        pred_scaled.append(p)
        seq = np.concatenate([seq[1:], p[None, :]], axis=0)

    predictions = scaler.inverse_transform(np.asarray(pred_scaled))

    # Align first forecast to last real observation
    predictions += df_hourly.iloc[-1].to_numpy(dtype=np.float32) - predictions[0]

# ---------------- Post-processing ----------------
# Clamp temp to plausible range
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)

# ---------------- Save to DB ----------------
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

print("✅ 24-hour forecast saved to SensorPrediction.")
