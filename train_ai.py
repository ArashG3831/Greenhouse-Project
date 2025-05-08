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

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()
from sensor.models import SensorData, SensorPrediction

# ---------------- CONFIG ----------------
MAX_LOOKBACK_STEPS   = 7 * 24 * 60      # max 7 days = 10,080
FORECAST_STEPS       = 24 * 60          # predict 1440 min (24h)
EPOCHS               = 3
BATCH_SIZE           = 4
MIN_SAMPLES          = 20

SENSOR_COLS = [
    "temperature", "humidity", "oxygen_level",
    "co2_level", "light_illumination"
]

# ---------------- LOAD & RESAMPLE ----------------
now = datetime.utcnow()
qs = SensorData.objects.filter(timestamp__gte=now - timedelta(days=7)).order_by("timestamp")
df = pd.DataFrame.from_records(qs.values("timestamp", *SENSOR_COLS))
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df = df.resample("1min").mean().dropna()

available_rows = len(df)
if available_rows < MIN_SAMPLES + 1:
    raise RuntimeError("❌ Not enough data.")

# ---------------- DYNAMIC LOOKBACK ----------------
lookback = min(MAX_LOOKBACK_STEPS, available_rows - MIN_SAMPLES)
print(f"DEBUG • lookback: {lookback} min, rows: {available_rows}")

# ---------------- SCALE + SEQUENCES ----------------
data = df[SENSOR_COLS].to_numpy(dtype=np.float32)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

X, Y = [], []
for i in range(len(scaled) - lookback):
    X.append(scaled[i:i+lookback])
    Y.append(scaled[i+lookback])
X, Y = np.array(X), np.array(Y)

# ---------------- BASELINE OR LSTM ----------------
if len(X) < MIN_SAMPLES:
    print("⚠️ Not enough training samples. Using baseline.")
    last_obs = df.iloc[-1].to_numpy()
    predictions = np.tile(last_obs, (FORECAST_STEPS, 1))
else:
    model = Sequential([
        Input(shape=(lookback, len(SENSOR_COLS))),
        LSTM(32, activation="tanh"),
        Dense(len(SENSOR_COLS))
    ])
    model.compile(optimizer=Adam(clipnorm=1.0), loss="mse")
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    seq = scaled[-lookback:].copy()
    pred_scaled = []
    for _ in range(FORECAST_STEPS):
        p = model.predict(seq[np.newaxis], verbose=0)[0]
        pred_scaled.append(p)
        seq = np.vstack([seq[1:], p])
    predictions = scaler.inverse_transform(np.array(pred_scaled))
    predictions += data[-1] - predictions[0]

# ---------------- CLAMP TEMP + SAVE HOURLY ----------------
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)
hourly = predictions[59::60][:24]
last_ts = df.index[-1]
timestamps = [last_ts + timedelta(hours=i+1) for i in range(24)]

records = [
    SensorPrediction(
        timestamp=ts,
        temperature=round(float(row[0]), 2),
        humidity=round(float(row[1]), 2),
        oxygen_level=round(float(row[2]), 2),
        co2_level=round(float(row[3]), 2),
        light_illumination=round(float(row[4]), 2)
    )
    for ts, row in zip(timestamps, hourly)
]

with transaction.atomic():
    SensorPrediction.objects.all().delete()
    SensorPrediction.objects.bulk_create(records, batch_size=24)

print("✅ Done. 24 hourly predictions saved.")
