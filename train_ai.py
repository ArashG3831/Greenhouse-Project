import os
import django
from datetime import timedelta

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input

# -- SETTINGS
LOOKBACK = 24
PREDICT_HOURS = 24

# -- DJANGO SETUP
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData, SensorPrediction

# -- LOAD DATA
qs = SensorData.objects.all().order_by("timestamp")
df = pd.DataFrame.from_records(qs.values(
    "timestamp", "temperature", "humidity", "oxygen_level", "co2_level", "light_illumination"
))

if df.empty:
    print("🚫 No data in database.")
    exit()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# -- RESAMPLE HOURLY
df_hourly = df.resample("h").mean().fillna(method="ffill").fillna(method="bfill")
sensor_cols = ["temperature", "humidity", "oxygen_level", "co2_level", "light_illumination"]

# -- USE MOST RECENT AVAILABLE DATA
df_hourly = df_hourly.tail(LOOKBACK + PREDICT_HOURS)
data = df_hourly[sensor_cols].values

# -- SCALE
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -- CREATE SEQUENCES
def create_sequences(dataset, lookback):
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i:i+lookback])
        Y.append(dataset[i+lookback])
    return np.array(X), np.array(Y)

X, Y = create_sequences(scaled_data, LOOKBACK)

# -- SPLIT OR FALLBACK
if len(X) < 1:
    print("⚠️ Not enough sequences. Using fallback repetition strategy.")
    fallback = data[-1]
    predictions = np.repeat(fallback[np.newaxis, :], PREDICT_HOURS, axis=0)
else:
    if len(X) < 5:
        X_train, Y_train = X, Y
        X_val, Y_val = X[:0], Y[:0]
    else:
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        Y_train, Y_val = Y[:split_idx], Y[split_idx:]

    if X_train.ndim != 3 or X_train.shape[1:] != (LOOKBACK, len(sensor_cols)):
        print(f"⚠️ Invalid training shape: {X_train.shape}. Using fallback.")
        fallback = data[-1]
        predictions = np.repeat(fallback[np.newaxis, :], PREDICT_HOURS, axis=0)
    else:
        # -- BUILD MODEL
        model = Sequential([
            Input(shape=(LOOKBACK, len(sensor_cols))),
            LSTM(64, activation='relu'),
            Dense(len(sensor_cols))
        ])
        model.compile(optimizer='adam', loss='mse')

        # -- TRAIN
        model.fit(X_train, Y_train,
                  validation_data=(X_val, Y_val) if len(X_val) > 0 else None,
                  epochs=30, batch_size=min(8, len(X_train)), verbose=1)

        # -- PREDICT
        last_seq = scaled_data[-LOOKBACK:]
        current_seq = last_seq.copy()
        predictions_scaled = []

        for _ in range(PREDICT_HOURS):
            pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0]
            predictions_scaled.append(pred)
            current_seq = np.vstack([current_seq[1:], pred])

        predictions_scaled = np.array(predictions_scaled)
        predictions = scaler.inverse_transform(predictions_scaled)

# -- CONTINUITY SHIFT
last_real = df_hourly.iloc[-1].values
predictions += (last_real - predictions[0])

# -- CLAMP TEMP
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)

# -- SAVE TO DB
SensorPrediction.objects.all().delete()
last_ts = df_hourly.index[-1]
timestamps = [last_ts + timedelta(hours=i+1) for i in range(PREDICT_HOURS)]

SensorPrediction.objects.bulk_create([
    SensorPrediction(
        timestamp=ts,
        temperature=round(row[0], 2),
        humidity=round(row[1], 2),
        oxygen_level=round(row[2], 2),
        co2_level=round(row[3], 2),
        light_illumination=round(row[4], 2)
    ) for ts, row in zip(timestamps, predictions)
])

print("✅ Trained on available data and saved 24h forecast.")
