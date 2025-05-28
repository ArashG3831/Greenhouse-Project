import random
import mysql.connector
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# --- Config ---
DB_CONFIG = {
    "host": "mysql",  # Change to "db" if inside Docker
    "user": "user",
    "password": "password",
    "database": "greenhouse",
}

# --- Time Setup ---
TEHRAN_TZ = ZoneInfo("Asia/Tehran")
NOW_TEHRAN = datetime.now(TEHRAN_TZ)
NOW_UTC = NOW_TEHRAN.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)  # <-- THIS is the key
START_UTC = NOW_UTC - timedelta(days=30)
INTERVAL = timedelta(seconds=5)
NUM_ENTRIES = int((NOW_UTC - START_UTC).total_seconds() / INTERVAL.total_seconds())

print(f"🔥 Generating {NUM_ENTRIES} entries from {START_UTC} to {NOW_UTC}")

# --- DB Connection ---
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# --- Insert Query ---
insert_query = """
    INSERT INTO sensor_sensordata
    (timestamp, temperature, humidity, soil_moisture, co2_level, light_illumination, leaf_color)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

# --- Initial values ---
temperature = random.uniform(25, 30)
humidity = random.uniform(60, 70)
co2 = random.uniform(400, 500)
light = random.uniform(600, 800)
soil_moisture = random.uniform(35, 50)
current_time = START_UTC

# --- Leaf color generator ---
def generate_leaf_color():
    green = random.randint(100, 255)
    red = random.randint(50, 180)
    blue = random.randint(0, 100)
    return f"#{red:02x}{green:02x}{blue:02x}"

# --- Batch insert ---
batch_size = 5000
buffer = []

for _ in range(NUM_ENTRIES):
    # Simulate fluctuations
    temperature += random.uniform(-0.05, 0.05)
    humidity += random.uniform(-0.15, 0.15)
    co2 += random.uniform(-0.1, 0.1)
    light += random.uniform(-0.5, 0.5)
    soil_moisture += random.uniform(-0.2, 0.2)

    # Clamp values
    temperature = max(20, min(temperature, 35))
    humidity = max(40, min(humidity, 90))
    co2 = max(300, min(co2, 600))
    light = max(200, min(light, 1200))
    soil_moisture = max(30, min(soil_moisture, 60))

    row = (
        current_time,
        round(temperature, 2),
        round(humidity, 2),
        round(soil_moisture, 2),
        round(co2, 2),
        round(light, 2),
        generate_leaf_color()
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
print("🎉 Done! All sensor data inserted.")
