from django.db import models

class SensorData(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    temperature = models.FloatField()
    humidity = models.FloatField()
    oxygen_level = models.FloatField()
    co2_level = models.FloatField(null=True, blank=True)
    light_illumination = models.FloatField()

    def __str__(self):
        return f"Temp: {self.temperature}°C, Humidity: {self.humidity}%, O2: {self.oxygen_level}%, CO2: {self.co2_level}, Light: {self.light_illumination}lx"

class ControlState(models.Model):
    """Stores the current state of Fan and Water controls."""
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
    # New fields:
    fan_is_running = models.BooleanField(default=False)
    last_water_dispense = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return (f"Fan: {self.fan_mode} (Running: {self.fan_is_running}), "
                f"Water: {self.water_mode}, Last Watered: {self.last_water_dispense}")
