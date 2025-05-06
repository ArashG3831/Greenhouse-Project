import os
import django
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

# --- Django setup ---
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData

# --- Config ---
TEHRAN_TZ = ZoneInfo("Asia/Tehran")
NOW = datetime.now(TEHRAN_TZ)
START_DATE = NOW - timedelta(days=60)
INTERVAL = timedelta(seconds=5)
BATCH_SIZE = 10000

# --- Compute number of entries ---
NUM_ENTRIES = int((NOW - START_DATE).total_seconds() / INTERVAL.total_seconds())
print(f"🔥 Generating {NUM_ENTRIES} entries from {START_DATE} to {NOW} in batches of {BATCH_SIZE}")

# --- Initial values ---
temperature = random.uniform(25, 30)
humidity = random.uniform(60, 70)
oxygen = random.uniform(19.5, 20.5)
co2 = random.uniform(400, 500)
light = random.uniform(600, 800)

current_time = START_DATE
batch = []
inserted = 0

for _ in range(NUM_ENTRIES):
    # Random walk simulation
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

    # Add to batch
    batch.append(SensorData(
        timestamp=current_time,
        temperature=round(temperature, 2),
        humidity=round(humidity, 2),
        oxygen_level=round(oxygen, 2),
        co2_level=round(co2, 2),
        light_illumination=round(light, 2),
    ))

    current_time += INTERVAL

    # Batch insert
    if len(batch) >= BATCH_SIZE:
        SensorData.objects.bulk_create(batch)
        inserted += len(batch)
        print(f"✅ Inserted {inserted}/{NUM_ENTRIES}")
        batch.clear()

# Final flush
if batch:
    SensorData.objects.bulk_create(batch)
    inserted += len(batch)
    print(f"✅ Final insert: {inserted}/{NUM_ENTRIES}")

print("🎉 All data inserted successfully.")
