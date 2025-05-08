#!/usr/bin/env python3
import os, django
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from django.db import transaction
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData, SensorPrediction

# ---------------- CONFIG ----------------
LOOKBACK_STEPS      = 60   # last 1 hour
FORECAST_STEPS      = 1440  # 24 hours (1-min intervals)
EPOCHS              = 3
BATCH_SIZE          = 4
MIN_TRAIN_SAMPLES   = 10

SENSOR_COLS = [
    "temperature", "humidity", "oxygen_level",
    "co2_level", "light_illumination"
]

# ---------------- LOAD + RESAMPLE ----------------
now = datetime.utcnow()
qs = SensorData.objects.filter(timestamp__gte=now - timedelta(days=1)).order_by("timestamp")
df = pd.DataFrame.from_records(qs.values("timestamp", *SENSOR_COLS))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df = df.resample("1min").mean().dropna()
print("DEBUG • 1-min rows:", len(df))

# ---------------- SCALE + SEQUENCES ----------------
data = df[SENSOR_COLS].to_numpy(dtype=np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

X, Y = [], []
for i in range(len(scaled) - LOOKBACK_STEPS):
    X.append(scaled[i:i+LOOKBACK_STEPS])
    Y.append(scaled[i+LOOKBACK_STEPS])
X, Y = np.array(X), np.array(Y)
print("DEBUG • X:", X.shape, "Y:", Y.shape)

# ---------------- BASELINE IF TOO LITTLE DATA ----------------
if len(X) < MIN_TRAIN_SAMPLES:
    print("⚠️ Not enough data – using last real value")
    last_real = df.iloc[-1].to_numpy()
    preds = np.tile(last_real, (FORECAST_STEPS, 1))
else:
    # ---------------- SIMPLE LSTM ----------------
    model = Sequential([
        Input(shape=(LOOKBACK_STEPS, len(SENSOR_COLS))),
        LSTM(32, activation="tanh"),
        Dense(len(SENSOR_COLS))
    ])
    model.compile(optimizer=Adam(clipnorm=1.0), loss="mse")
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    seq = scaled[-LOOKBACK_STEPS:].copy()
    pred_scaled = []
    for _ in range(FORECAST_STEPS):
        p = model.predict(seq[np.newaxis], verbose=0)[0]
        pred_scaled.append(p)
        seq = np.vstack([seq[1:], p])
    preds = scaler.inverse_transform(np.array(pred_scaled))
    preds += data[-1] - preds[0]  # align first prediction

# ---------------- CLAMP + DOWNSAMPLE HOURLY ----------------
preds[:, 0] = np.clip(preds[:, 0], 15, 40)
hourly_preds = preds[59::60][:24]
last_ts = df.index[-1]
timestamps = [last_ts + timedelta(hours=i+1) for i in range(len(hourly_preds))]

records = [
    SensorPrediction(
        timestamp=ts,
        temperature=round(float(row[0]), 2),
        humidity=round(float(row[1]), 2),
        oxygen_level=round(float(row[2]), 2),
        co2_level=round(float(row[3]), 2),
        light_illumination=round(float(row[4]), 2)
    )
    for ts, row in zip(timestamps, hourly_preds)
]

with transaction.atomic():
    SensorPrediction.objects.all().delete()
    SensorPrediction.objects.bulk_create(records, batch_size=24)

print("✅ Wrote 24 hourly predictions.")
