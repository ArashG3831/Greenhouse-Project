import os
import django
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

from sensor.models import SensorData

# --- Setup ---
TEHRAN_TZ = ZoneInfo("Asia/Tehran")
NOW = datetime.now(TEHRAN_TZ)
START_DATE = NOW - timedelta(days=60)
INTERVAL = timedelta(seconds=5)

NUM_ENTRIES = int((NOW - START_DATE).total_seconds() / INTERVAL.total_seconds())

print(f"🔥 Generating {NUM_ENTRIES} entries from {START_DATE} to {NOW}")

# --- Baseline Ranges ---
temperature = random.uniform(25, 30)
humidity = random.uniform(60, 70)
oxygen = random.uniform(19.5, 20.5)
co2 = random.uniform(400, 500)
light = random.uniform(600, 800)

current_time = START_DATE
batch = []

# --- Generate ---
for _ in range(NUM_ENTRIES):
    temperature += random.uniform(-0.05, 0.05)
    humidity += random.uniform(-0.15, 0.15)
    oxygen += random.uniform(-0.005, 0.005)
    co2 += random.uniform(-0.1, 0.1)
    light += random.uniform(-0.5, 0.5)

    temperature = max(20, min(temperature, 35))
    humidity = max(40, min(humidity, 90))
    oxygen = max(19, min(oxygen, 21))
    co2 = max(300, min(co2, 600))
    light = max(200, min(light, 1200))

    batch.append(SensorData(
        timestamp=current_time,
        temperature=round(temperature, 2),
        humidity=round(humidity, 2),
        oxygen_level=round(oxygen, 2),
        co2_level=round(co2, 2),
        light_illumination=round(light, 2),
    ))

    current_time += INTERVAL

# --- Insert in bulk ---
SensorData.objects.bulk_create(batch, batch_size=1000)
print("✅ Successfully generated and inserted 60 days of sensor data!")