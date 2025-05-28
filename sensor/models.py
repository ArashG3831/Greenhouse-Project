from django.db import models
from django.utils import timezone


class SensorPrediction(models.Model):
    timestamp = models.DateTimeField()
    temperature = models.FloatField()
    humidity = models.FloatField()
    soil_moisture = models.FloatField()
    co2_level = models.FloatField()
    light_illumination = models.FloatField()

    def __str__(self):
        return f"[Prediction @ {self.timestamp}] T={self.temperature} H={self.humidity} Soil={self.soil_moisture}"


class SensorData(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    temperature = models.FloatField()
    humidity = models.FloatField()
    co2_level = models.FloatField(null=True, blank=True)
    light_illumination = models.FloatField()
    soil_moisture = models.FloatField()

    # ✅ Default value added to shut Django up forever
    leaf_color = models.CharField(max_length=20, default="#AABBCC")

    def __str__(self):
        return (f"Temp: {self.temperature}°C, Humidity: {self.humidity}%, "
                f"Soil Moisture: {self.soil_moisture}%, CO2: {self.co2_level}, "
                f"Light: {self.light_illumination}lx, Leaf: {self.leaf_color}")


class ControlState(models.Model):
    fan_mode = models.CharField(
        max_length=10,
        choices=[("auto", "Auto"), ("on", "On"), ("off", "Off")],
        default="auto"
    )
    water_mode = models.CharField(
        max_length=10,
        choices=[("auto", "Auto"), ("on", "On"), ("off", "Off")],
        default="auto"
    )
    fan_is_running = models.BooleanField(default=False)
    last_water_dispense = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return (f"Fan: {self.fan_mode} (Running: {self.fan_is_running}), "
                f"Water: {self.water_mode}, Last Watered: {self.last_water_dispense}")
