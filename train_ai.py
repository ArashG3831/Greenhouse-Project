# ---------------------------------------------------------------------
# 8. DOWNSAMPLE TO HOURLY PREDICTIONS + SAVE
# ---------------------------------------------------------------------
last_ts = df.index[-1]

# pick every 60th row from 1-minute forecast
selected_predictions = predictions[59::60][:24]  # 59, 119, ..., 1439

timestamps = [
    last_ts + timedelta(hours=i + 1) for i in range(len(selected_predictions))
]

records = [
    SensorPrediction(
        timestamp=ts,
        temperature=round(float(r[0]), 2),
        humidity=round(float(r[1]), 2),
        oxygen_level=round(float(r[2]), 2),
        co2_level=round(float(r[3]), 2),
        light_illumination=round(float(r[4]), 2),
    )
    for ts, r in zip(timestamps, selected_predictions)
]

with transaction.atomic():
    SensorPrediction.objects.all().delete()
    SensorPrediction.objects.bulk_create(records, batch_size=24)

print(f"✅ Wrote {len(records)} hourly prediction rows to SensorPrediction")
