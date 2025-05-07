import random
import mysql.connector
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# --- Config ---
DB_CONFIG = {
    "host": "localhost",  # or '127.0.0.1' if you're outside Docker
    "user": "user",
    "password": "password",
    "database": "greenhouse",
}

TEHRAN_TZ = ZoneInfo("Asia/Tehran")
NOW = datetime.now(TEHRAN_TZ)
START_DATE = NOW - timedelta(days=60)
INTERVAL = timedelta(seconds=5)
NUM_ENTRIES = int((NOW - START_DATE).total_seconds() / INTERVAL.total_seconds())

print(f"🔥 Generating {NUM_ENTRIES} entries from {START_DATE} to {NOW}")

# --- DB Connection ---
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# --- Prepare Insert Query ---
insert_query = """
    INSERT INTO sensor_sensordata 
    (timestamp, temperature, humidity, oxygen_level, co2_level, light_illumination)
    VALUES (%s, %s, %s, %s, %s, %s)
"""

# --- Baseline values ---
temperature = random.uniform(25, 30)
humidity = random.uniform(60, 70)
oxygen = random.uniform(19.5, 20.5)
co2 = random.uniform(400, 500)
light = random.uniform(600, 800)
current_time = START_DATE

# --- Batch insert ---
batch_size = 5000
buffer = []

for i in range(NUM_ENTRIES):
    temperature += random.uniform(-0.05, 0.05)
    humidity += random.uniform(-0.15, 0.15)
    oxygen += random.uniform(-0.005, 0.005)
    co2 += random.uniform(-0.1, 0.1)
    light += random.uniform(-0.5, 0.5)

    # Clamp values
    temperature = max(20, min(temperature, 35))
    humidity = max(40, min(humidity, 90))
    oxygen = max(19, min(oxygen, 21))
    co2 = max(300, min(co2, 600))
    light = max(200, min(light, 1200))

    row = (
        current_time.strftime("%Y-%m-%d %H:%M:%S"),
        round(temperature, 2),
        round(humidity, 2),
        round(oxygen, 2),
        round(co2, 2),
        round(light, 2),
    )
    buffer.append(row)
    current_time += INTERVAL

    if len(buffer) >= batch_size:
        cursor.executemany(insert_query, buffer)
        conn.commit()
        print(f"✅ Inserted {len(buffer)} rows...")
        buffer = []

# Final flush
if buffer:
    cursor.executemany(insert_query, buffer)
    conn.commit()
    print(f"✅ Inserted final {len(buffer)} rows...")

cursor.close()
conn.close()
print("✅ All done!")
