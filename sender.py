import sqlite3
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# Path to your SQLite database
DB_FILE = "db.sqlite3"

# Connect to SQLite
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Define initial “baseline” values (typical greenhouse ranges)
temperature = random.uniform(25, 30)
humidity = random.uniform(60, 70)
oxygen = random.uniform(19.5, 20.5)
co2 = random.uniform(400, 500)
light = random.uniform(600, 800)

# Set Tehran timezone
TEHRAN_TZ = ZoneInfo("Asia/Tehran")

print("Starting continuous data insertion (one entry every 5 seconds)...")

while True:
    # Make small random changes (random walk)
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

    # Prepare the data tuple (timestamp in Tehran time)
    data = (
        datetime.now(TEHRAN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        round(temperature, 2),
        round(humidity, 2),
        round(oxygen, 2),
        round(co2, 2),
        round(light, 2)
    )

    try:
        cursor.execute("""
            INSERT INTO sensor_sensordata 
            (timestamp, temperature, humidity, oxygen_level, co2_level, light_illumination)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)
        conn.commit()
        print(f"✅ Inserted: {data}")
    except Exception as e:
        print("❌ Failed to insert data:", e)

    time.sleep(5)  # Wait 5 seconds before generating the next entry
