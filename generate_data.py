import sqlite3
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # For timezone support (Python 3.9+)

DB_FILE = "db.sqlite3"

# Set Tehran timezone
TEHRAN_TZ = ZoneInfo("Asia/Tehran")

# Get current time in Tehran and calculate start date (60 days ago)
NOW = datetime.now(TEHRAN_TZ)
START_DATE = NOW - timedelta(days=60)
INTERVAL = timedelta(seconds=5)

# Connect to SQLite
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Calculate how many entries fit up until now
NUM_ENTRIES = int((NOW - START_DATE).total_seconds() / INTERVAL.total_seconds())

print(f"🔥 Generating {NUM_ENTRIES} entries from {START_DATE} to {NOW}")

# --- 1) Define initial “baseline” values within normal greenhouse ranges ---
temperature = random.uniform(25, 30)      # Typical range ~20–35°C
humidity = random.uniform(60, 70)         # Typical range ~40–90%
oxygen = random.uniform(19.5, 20.5)         # Typical range ~19–21%
co2 = random.uniform(400, 500)              # Typical range ~300–600 ppm
light = random.uniform(600, 800)            # Typical range ~200–1200 lx

current_time = START_DATE

# --- 2) Insert data in a loop, making small random “nudges” each time ---
for _ in range(NUM_ENTRIES):
    # Adjust values slightly (random walk)
    temperature += random.uniform(-0.05, 0.05)
    humidity += random.uniform(-0.15, 0.15)
    oxygen += random.uniform(-0.005, 0.005)
    co2 += random.uniform(-0.1, 0.1)
    light += random.uniform(-0.5, 0.5)

    # Constrain values to realistic ranges
    temperature = max(20, min(temperature, 35))
    humidity = max(40, min(humidity, 90))
    oxygen = max(19, min(oxygen, 21))
    co2 = max(300, min(co2, 600))
    light = max(200, min(light, 1200))

    # Prepare data with timestamp in Tehran time (ISO format with timezone)
    data = (
        current_time.isoformat(),  # e.g., "2025-03-18T22:28:22.406161+03:30"
        round(temperature, 2),
        round(humidity, 2),
        round(oxygen, 2),
        round(co2, 2),
        round(light, 2)
    )

    # Insert the row into the sensor_sensordata table
    cursor.execute("""
        INSERT INTO sensor_sensordata (timestamp, temperature, humidity, oxygen_level, co2_level, light_illumination)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data)

    # Move forward 5 seconds
    current_time += INTERVAL

conn.commit()
conn.close()

print("✅ 60 days of *continuous* sensor data generated successfully (UP TO NOW)!")
