import sqlite3
import random
from datetime import datetime, timedelta

DB_FILE = "db.sqlite3"
NOW = datetime.now()
START_DATE = NOW - timedelta(days=60)
INTERVAL = timedelta(seconds=5)

# Connect to SQLite
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Calculate how many entries fit up until now
NUM_ENTRIES = int((NOW - START_DATE).total_seconds() / INTERVAL.total_seconds())

print(f"🔥 Generating {NUM_ENTRIES} entries from {START_DATE} to {NOW}")

# --- 1) Define initial “baseline” values within normal greenhouse ranges ---
temperature = random.uniform(25, 30)      # Typical greenhouse range ~20–35
humidity = random.uniform(60, 70)         # Typical greenhouse range ~40–90
oxygen = random.uniform(19.5, 20.5)       # Typical range ~19–21
co2 = random.uniform(400, 500)            # Typical range ~300–600
light = random.uniform(600, 800)          # Typical range ~200–1200

current_time = START_DATE

# --- 2) Insert data in a loop, doing small random “nudges” each time ---
for _ in range(NUM_ENTRIES):
    # Make small random changes (random walk) around the current value
    temperature += random.uniform(-0.05, 0.05)  # tweak for slow drift
    humidity += random.uniform(-0.15, 0.15)
    oxygen += random.uniform(-0.005, 0.005)
    co2 += random.uniform(-0.1, 0.1)
    light += random.uniform(-0.5, 0.5)

    # Constrain values to realistic greenhouse ranges
    temperature = max(20, min(temperature, 35))
    humidity = max(40, min(humidity, 90))
    oxygen = max(19, min(oxygen, 21))
    co2 = max(300, min(co2, 600))
    light = max(200, min(light, 1200))

    # Round values for neatness
    data = (
        current_time.strftime("%Y-%m-%d %H:%M:%S"),
        round(temperature, 2),
        round(humidity, 2),
        round(oxygen, 2),
        round(co2, 2),
        round(light, 2)
    )

    # Insert the row
    cursor.execute("""
        INSERT INTO sensor_sensordata (timestamp, temperature, humidity, oxygen_level, co2_level, light_illumination)
        VALUES (?, ?, ?, ?, ?, ?)
    """, data)

    current_time += INTERVAL  # move forward 5 seconds

conn.commit()
conn.close()

print("✅ 60 days of *continuous* sensor data generated successfully (UP TO NOW)!")
