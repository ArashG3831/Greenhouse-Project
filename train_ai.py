import os
import django
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData, SensorPrediction

# Parameters
FORECAST_STEPS = 24     # predict 24 hourly steps
LOOKBACK_MINUTES = 60 * 24 * 7  # 7 days max
RESAMPLE_INTERVAL = '1min'
BATCH_SIZE = 8
EPOCHS = 3

print("⏳ Loading data from DB...")
now = datetime.now()
cutoff = now - timedelta(minutes=LOOKBACK_MINUTES)

qs = SensorData.objects.filter(timestamp__gte=cutoff).order_by("timestamp")
df = pd.DataFrame.from_records(qs.values("timestamp", "temperature", "humidity", "oxygen_level", "co2_level", "light_illumination"))

if df.empty or len(df) < 100:
    raise ValueError("❌ Not enough data to train. Got only %d rows." % len(df))

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# Downsample to 1 minute
df = df.resample(RESAMPLE_INTERVAL).mean().interpolate()

sensor_cols = ["temperature", "humidity", "oxygen_level", "co2_level", "light_illumination"]
data = df[sensor_cols].to_numpy()

# Scale
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_seq2seq_data(data, input_len, output_len):
    X, Y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        Y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(Y)

X, Y = create_seq2seq_data(scaled_data, len(df) - FORECAST_STEPS, FORECAST_STEPS)

print(f"DEBUG • X shape: {X.shape}, Y shape: {Y.shape}")

# Build GRU Sequence-to-Sequence Model
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    GRU(64, activation="tanh"),
    RepeatVector(FORECAST_STEPS),
    GRU(64, activation="tanh", return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])

model.compile(optimizer=Adam(0.001), loss='mse')
model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Predict the next 24 hours
last_seq = scaled_data[-X.shape[1]:]
pred_scaled = model.predict(np.array([last_seq]))[0]
pred = scaler.inverse_transform(pred_scaled)

# Get hourly timestamps
last_ts = df.index[-1]
hourly_preds = pred[::60]  # 60-min step from 1-min predictions
timestamps = [last_ts + timedelta(hours=i + 1) for i in range(FORECAST_STEPS)]

# Clamp temperature
hourly_preds[:, 0] = np.clip(hourly_preds[:, 0], 15, 40)

# Save
SensorPrediction.objects.all().delete()
SensorPrediction.objects.bulk_create([
    SensorPrediction(
        timestamp=ts,
        temperature=round(row[0], 2),
        humidity=round(row[1], 2),
        oxygen_level=round(row[2], 2),
        co2_level=round(row[3], 2),
        light_illumination=round(row[4], 2),
    )
    for ts, row in zip(timestamps, hourly_preds)
])

print("✅ Done. 24 hourly predictions saved.")
