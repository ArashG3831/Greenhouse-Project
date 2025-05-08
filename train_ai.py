import os
import django
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

# -- 0) SETTINGS
LOOKBACK = 24

# -- 1) DJANGO SETUP
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData, SensorPrediction

# -- 2) LOAD DATA FROM DB
qs = SensorData.objects.all().order_by("timestamp")
df = pd.DataFrame.from_records(qs.values(
    "timestamp", "temperature", "humidity", "oxygen_level", "co2_level", "light_illumination"
))

# -- 3) PREPROCESSING
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# -- 4) RESAMPLE HOURLY, FILL GAPS, SELECT LATEST WINDOW
df_hourly = df.resample("h").mean()
df_hourly = df_hourly.fillna(method="ffill").fillna(method="bfill")
df_hourly = df_hourly.tail(LOOKBACK + 24)

if len(df_hourly) < LOOKBACK + 1:
    raise ValueError(f"🚫 Not enough hourly data to form a single sequence. Got {len(df_hourly)}, need at least {LOOKBACK + 1} hours.")

sensor_cols = ["temperature", "humidity", "oxygen_level", "co2_level", "light_illumination"]
data = df_hourly[sensor_cols].values

# -- 5) SCALE DATA
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -- 6) CREATE SEQUENCES
def create_sequences(dataset, lookback=24):
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i : i + lookback])
        Y.append(dataset[i + lookback])
    return np.array(X), np.array(Y)

X, Y = create_sequences(scaled_data, LOOKBACK)

if len(X) == 0 or X.ndim != 3:
    raise ValueError(f"⚠️ Not enough data to train. Required shape: (samples, {LOOKBACK}, {len(sensor_cols)}), got: {X.shape}")

# -- 7) TRAIN/VAL SPLIT
# -- 7) TRAIN/VAL SPLIT
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]

if X_train.shape[0] < 1 or X_train.ndim != 3 or X_train.shape[1:] != (LOOKBACK, len(sensor_cols)):
    print(f"⚠️ Skipping training due to insufficient or invalid shape: {X_train.shape}")
    predictions = np.repeat(data[-1][np.newaxis, :], 24, axis=0)

else:
    # -- 8) BUILD MODEL
    model = Sequential([
        Input(shape=(LOOKBACK, len(sensor_cols))),
        LSTM(64, activation='relu'),
        Dense(len(sensor_cols))
    ])
    model.compile(optimizer='adam', loss='mse')

    # -- 9) TRAIN
    batch_size = min(8, len(X_train))
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val) if len(X_val) > 0 else None,
        epochs=30,
        batch_size=batch_size,
        verbose=1
    )

    # -- 10) PREDICT NEXT 24 HOURS
    last_24 = scaled_data[-LOOKBACK:]
    current_seq = last_24.copy()
    predictions_scaled = []

    for _ in range(24):
        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0]
        predictions_scaled.append(pred)
        current_seq = np.vstack([current_seq[1:], pred])

    predictions_scaled = np.array(predictions_scaled)
    predictions = scaler.inverse_transform(predictions_scaled)

# -- 11) CONTINUITY SHIFT
last_real = df_hourly.iloc[-1].values
diff = last_real - predictions[0]
predictions += diff

# -- 12) CLAMP TEMPERATURE
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)

# -- 13) SAVE TO DB
SensorPrediction.objects.all().delete()

last_ts = df_hourly.index[-1]
timestamps = [last_ts + timedelta(hours=i+1) for i in range(24)]

records = [
    SensorPrediction(
        timestamp=ts,
        temperature=round(row[0], 2),
        humidity=round(row[1], 2),
        oxygen_level=round(row[2], 2),
        co2_level=round(row[3], 2),
        light_illumination=round(row[4], 2)
    )
    for ts, row in zip(timestamps, predictions)
]

SensorPrediction.objects.bulk_create(records, batch_size=24)

print("✅ Predictions saved in MySQL via Django ORM (24h forecast).")
