import sqlite3
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# -- 1) CONNECT TO THE DB & LOAD DATA
DB_PATH = "db.sqlite3"
conn = sqlite3.connect(DB_PATH)

query = """
    SELECT timestamp, temperature, humidity, oxygen_level, co2_level, light_illumination
    FROM sensor_sensordata
"""
df = pd.read_sql_query(query, conn)
df["timestamp"] = pd.to_datetime(df["timestamp"], format='ISO8601')
df.sort_values(by="timestamp", inplace=True)
df.set_index("timestamp", inplace=True)

# -- 2) RESAMPLE HOURLY & FILTER LAST 7 DAYS
df_hourly = df.resample("h").mean().dropna()
seven_days_ago = df_hourly.index.max() - pd.Timedelta(days=7)
df_hourly = df_hourly.loc[df_hourly.index >= seven_days_ago]

sensor_cols = ["temperature", "humidity", "oxygen_level", "co2_level", "light_illumination"]
data = df_hourly[sensor_cols].values  # shape: (num_hours, 5)

# -- 3) SCALE DATA
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -- 4) CREATE SEQUENCES (LOOKBACK=24 HOURS)
LOOKBACK = 24
def create_sequences(dataset, lookback=24):
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i : i + lookback])  # last 24 hours
        Y.append(dataset[i + lookback])      # next hour (one-step)
    return np.array(X), np.array(Y)

X, Y = create_sequences(scaled_data, LOOKBACK)

# -- 5) SPLIT INTO TRAIN/VAL
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]

# -- 6) BUILD & TRAIN LSTM
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(LOOKBACK, len(sensor_cols))))
model.add(Dense(len(sensor_cols)))  # outputs 5 sensor values
model.compile(optimizer='adam', loss='mse')

model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30,
    batch_size=8,
    verbose=1
)

# -- 7) ITERATIVE FORECAST FOR 24 HOURS
last_24 = scaled_data[-LOOKBACK:]  # shape: (24, 5)
current_seq = last_24.copy()

predictions_scaled = []
for i in range(24):
    pred = model.predict(current_seq[np.newaxis, :, :])[0]  # shape: (5,)
    predictions_scaled.append(pred)
    # shift the window by 1 hour
    current_seq = np.vstack([current_seq[1:], pred])

predictions_scaled = np.array(predictions_scaled)
predictions = scaler.inverse_transform(predictions_scaled)  # shape: (24, 5)

# -- 8) FORCE CONTINUITY: SHIFT FIRST PREDICTION TO MATCH LAST REAL DATA
last_real = df_hourly.iloc[-1].values  # shape: (5,)
diff = last_real - predictions[0]      # shape: (5,)
predictions = predictions + diff       # shift entire 24-hr forecast

# -- 9) CLAMP TEMPERATURE BETWEEN 15°C AND 40°C
# sensor_cols = ["temperature", "humidity", "oxygen_level", "co2_level", "light_illumination"]
# Index 0 in each row is temperature
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)

# (OPTIONAL) If you want to clamp other variables, do similarly:
# e.g. clamp humidity in [40, 90], etc.

# -- 10) SAVE FORECAST TO DB
last_timestamp = df_hourly.index[-1]
future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(24)]
pred_df = pd.DataFrame(predictions, columns=sensor_cols)
pred_df["timestamp"] = future_timestamps

pred_df.to_sql("sensor_predictions", conn, if_exists="replace", index=False)
conn.commit()
conn.close()

# -- 11) PLOT THE LAST 7 DAYS + 24-HOUR FORECAST
# plt.figure(figsize=(12, 6))
#
# seven_days_ago = df_hourly.index.max() - pd.Timedelta(days=7)
# hist_df = df_hourly.loc[df_hourly.index >= seven_days_ago].reset_index()
#
# plt.plot(hist_df["timestamp"], hist_df["temperature"], label="Past Temperature", color="blue")
# plt.plot(pred_df["timestamp"], pred_df["temperature"], label="Predicted Temperature (Next 24h)",
#          color="red", linestyle="dashed")
#
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
# plt.xticks(rotation=45)
# plt.xlabel("Time")
# plt.ylabel("Temperature (°C)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# print("✅ Done! Predictions stored in DB, with forced continuity and temperature clamping.")
