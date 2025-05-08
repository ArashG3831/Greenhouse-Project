"""
train_ai.py – robust even with just a handful of hourly rows
"""

import os, django
from datetime import timedelta

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

# ─── CONSTANTS ──────────────────────────────────────────────────────────────────
LOOKBACK       = 24          # hours per input window
PREDICT_HOURS  = 24          # hours to forecast
SENSOR_COLS    = ["temperature", "humidity",
                  "oxygen_level", "co2_level", "light_illumination"]

# ─── DJANGO SET-UP ─────────────────────────────────────────────────────────────
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData, SensorPrediction

# ─── LOAD & PRE-PROCESS ────────────────────────────────────────────────────────
qs = SensorData.objects.all().order_by("timestamp")
df = pd.DataFrame.from_records(
         qs.values("timestamp", *SENSOR_COLS) )

if df.empty:
    print("🚫  No sensor data in DB – nothing to train.")
    exit()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# hourly mean, forward/backward fill gaps
df_h = (df.resample("h").mean()
          .fillna(method="ffill")
          .fillna(method="bfill"))

# keep only what we need
df_h = df_h.tail(LOOKBACK + PREDICT_HOURS)
data  = df_h[SENSOR_COLS].values           # shape: n_hours × 5

# ─── SCALING ───────────────────────────────────────────────────────────────────
scaler      = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ─── BUILD SEQUENCES ───────────────────────────────────────────────────────────
def build_sequences(arr, lookback):
    """Return X (n_seq×lookback×feat)  and  Y (n_seq×feat)."""
    seqs, targets = [], []
    for i in range(len(arr) - lookback):
        seqs.append(arr[i:i+lookback])
        targets.append(arr[i+lookback])
    if not seqs:                            # zero sequences
        return np.empty((0, lookback, len(SENSOR_COLS))), np.empty((0,len(SENSOR_COLS)))
    # Force proper stacking – raises if shapes differ
    return np.stack(seqs).astype("float32"), np.stack(targets).astype("float32")

try:
    X, Y = build_sequences(scaled_data, LOOKBACK)
except ValueError as e:                     # variable-length rows → fallback
    print(f"⚠️  Irregular sequence shapes ({e}).  Falling back.")
    X, Y = np.empty((0,LOOKBACK,len(SENSOR_COLS))), np.empty((0,len(SENSOR_COLS)))

# ───  Fallback if we still cannot train ────────────────────────────────────────
if X.shape[0] < 1:
    print("⚠️  Not enough clean sequences – repeating last value for forecast.")
    predictions = np.repeat(data[-1][None, :], PREDICT_HOURS, axis=0)

else:
    # ─── SIMPLE LSTM  ──────────────────────────────────────────────────────────
    model = Sequential([
        Input(shape=(LOOKBACK, len(SENSOR_COLS))),
        LSTM(64, activation="relu"),
        Dense(len(SENSOR_COLS))
    ])
    model.compile(optimizer="adam", loss="mse")

    model.fit(X, Y,
              epochs=30,
              batch_size=min(8, X.shape[0]),
              verbose=1)

    # ─── AUTOREGRESSIVE FORECAST  ─────────────────────────────────────────────
    seq = scaled_data[-LOOKBACK:]           # last 24 h, shape (24×5)
    preds_scaled = []
    for _ in range(PREDICT_HOURS):
        p = model.predict(seq[None, :, :], verbose=0)[0]   # 1×5 → 5
        preds_scaled.append(p)
        seq = np.vstack([seq[1:], p])       # slide window

    predictions = scaler.inverse_transform(np.array(preds_scaled))

# ─── CONTINUITY & CLAMP ────────────────────────────────────────────────────────
shift = df_h.iloc[-1].values - predictions[0]
predictions += shift
predictions[:,0] = np.clip(predictions[:,0], 15, 40)       # temp

# ─── SAVE 24-H FORECAST ────────────────────────────────────────────────────────
SensorPrediction.objects.all().delete()
start_ts = df_h.index[-1]
rows = [
    SensorPrediction(
        timestamp=start_ts + timedelta(hours=i+1),
        temperature       = round(r[0], 2),
        humidity          = round(r[1], 2),
        oxygen_level      = round(r[2], 2),
        co2_level         = round(r[3], 2),
        light_illumination= round(r[4], 2),
    )
    for i, r in enumerate(predictions)
]
SensorPrediction.objects.bulk_create(rows, batch_size=len(rows))

print("✅ 24-hour forecast generated and saved.")
