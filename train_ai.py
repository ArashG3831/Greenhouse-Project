import os
import django
import numpy as np
import pandas as pd
from datetime import timedelta
from django.utils.timezone import now as django_now

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Input
from tensorflow.keras.optimizers import Adam

# --- Django setup ---
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData, SensorPrediction

# --- Parameters ---
FORECAST_HOURS = 24
INPUT_MINUTES = 60 * 24 * 7  # 7 days = 10080 minutes
RESAMPLE_INTERVAL = '1min'
EPOCHS = 3
BATCH_SIZE = 8

print("⏳ Loading and preprocessing data...")
now = django_now()
cutoff = now - timedelta(minutes=INPUT_MINUTES)

# --- Fetch last 7 days of data ---
qs = SensorData.objects.filter(timestamp__gte=cutoff).order_by("timestamp")
df = pd.DataFrame.from_records(qs.values(
    "timestamp", "temperature", "humidity", "soil_moisture", "co2_level", "light_illumination"
))

if df.empty or len(df) < 1000:
    raise ValueError(f"❌ Not enough data to train. Got only {len(df)} rows.")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.sort_index(inplace=True)

# --- Resample to 1-minute intervals ---
df = df.resample(RESAMPLE_INTERVAL).mean().interpolate()

sensor_cols = ["temperature", "humidity", "soil_moisture", "co2_level", "light_illumination"]
data = df[sensor_cols].to_numpy()

# --- Normalize ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# --- Prepare sequences for training ---
input_steps = len(scaled_data) - FORECAST_HOURS  # input = everything except last 24
X = np.array([scaled_data[:input_steps]])
Y = np.array([scaled_data[input_steps:input_steps + FORECAST_HOURS]])

# --- Build GRU Model ---
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    GRU(64, activation="tanh"),
    RepeatVector(FORECAST_HOURS),
    GRU(64, activation="tanh", return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])

model.compile(optimizer=Adam(0.001), loss='mse')
model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# --- Predict the next 24 hours ---
pred_scaled = model.predict(X)[0]
pred = scaler.inverse_transform(pred_scaled)

# --- Create timestamps: 1 point per hour ---
last_ts = df.index[-1]
timestamps = [last_ts + timedelta(hours=i + 1) for i in range(FORECAST_HOURS)]

# --- Save predictions to DB ---
SensorPrediction.objects.all().delete()

SensorPrediction.objects.bulk_create([
    SensorPrediction(
        timestamp=ts,
        temperature=round(row[0], 2),
        humidity=round(row[1], 2),
        soil_moisture=round(row[2], 2),
        co2_level=round(row[3], 2),
        light_illumination=round(row[4], 2),
    )
    for ts, row in zip(timestamps, pred[::60])  # 1 point per hour from 1-min data
])

print("✅ Training complete. 24 hourly predictions saved.")
