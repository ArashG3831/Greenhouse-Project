from django.http import JsonResponse
import sqlite3
import pandas as pd
from django.db.models import Avg, F, Min, ExpressionWrapper, IntegerField
from django.utils.timezone import now, timedelta
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import SensorData, ControlState  # ✅ Imported ControlState!


@api_view(['GET'])
def get_data(request):
    """Fetch sensor data with correct timestamps for downsampling, ensuring the latest entry is always included."""
    time_range = request.GET.get('range', '7d')  # 🔥 Default to Last 7 Days

    # ✅ Define downsampling factors (Reduce data points)
    if time_range == '1h':
        start_date = now() - timedelta(hours=1)
        group_factor = 36
    elif time_range == '24h':
        start_date = now() - timedelta(hours=24)
        group_factor = 720
    elif time_range == '7d':
        start_date = now() - timedelta(days=7)
        group_factor = 720
    elif time_range == '30d':
        start_date = now() - timedelta(days=30)
        group_factor = 2880
    else:  # ✅ "All" case
        start_date = None
        group_factor = 2880

    # ✅ Fetch & Downsample Data
    if start_date:
        data = list(
            SensorData.objects.filter(timestamp__gte=start_date)
            .annotate(
                group_id=ExpressionWrapper(F("id") / group_factor, output_field=IntegerField())
            )
            .values("group_id")
            .annotate(
                timestamp=Min("timestamp"),  # ✅ Pick the earliest timestamp per group
                temperature=Avg("temperature"),
                humidity=Avg("humidity"),
                oxygen_level=Avg("oxygen_level"),
                co2_level=Avg("co2_level"),
                light_illumination=Avg("light_illumination"),
            )
            .order_by("timestamp")
        )
    else:
        data = list(
            SensorData.objects.all()
            .annotate(
                group_id=ExpressionWrapper(F("id") / group_factor, output_field=IntegerField())
            )
            .values("group_id")
            .annotate(
                timestamp=Min("timestamp"),  # ✅ Pick earliest timestamp per group
                temperature=Avg("temperature"),
                humidity=Avg("humidity"),
                oxygen_level=Avg("oxygen_level"),
                co2_level=Avg("co2_level"),
                light_illumination=Avg("light_illumination"),
            )
            .order_by("timestamp")
        )

    # ✅ Always Append the Latest Entry (Ensures Last Recorded Data Is Kept)
    latest_entry = (
        SensorData.objects.order_by("-timestamp")
        .values("timestamp", "temperature", "humidity", "oxygen_level", "co2_level", "light_illumination")
        .first()
    )
    if latest_entry and (not data or latest_entry["timestamp"] != data[-1]["timestamp"]):
        data.append(latest_entry)  # ✅ Append latest data if it's missing

    return Response(data)


@api_view(['POST'])
def set_control_state(request):
    """Updates the control mode for Fan & Water, with a one-time +10ml water dispense option."""
    control, _ = ControlState.objects.get_or_create(id=1)

    # Update fan mode if provided
    fan_mode = request.data.get("fan_mode")
    if fan_mode in ["auto", "on", "off"]:
        control.fan_mode = fan_mode

    # Handle water control update
    water_mode = request.data.get("water_mode")
    if water_mode in ["auto", "off"]:
        control.water_mode = water_mode
    elif water_mode == "+10ml":
        # Trigger water dispense: update the last water dispense timestamp.
        control.last_water_dispense = timezone.now()
        # Optionally, you may want to reset the water_mode to "off" or keep it as "auto"
        # control.water_mode = "off"  # for example

    control.save()

    return Response({
        "message": "Control state updated!",
        "fan_mode": control.fan_mode,
        "water_mode": control.water_mode,
        "last_water_dispense": control.last_water_dispense
    })

@api_view(['GET'])
def get_control_state(request):
    """Returns the current control mode for Fan & Water along with the last water dispense time."""
    control, _ = ControlState.objects.get_or_create(id=1)
    return Response({
        "fan_mode": control.fan_mode,
        "water_mode": control.water_mode,
        "last_water_dispense": control.last_water_dispense
    })


@api_view(['POST'])
def receive_data(request):
    """Receives sensor data from sender.py and saves it to the database."""
    try:
        data = request.data
        print("🔥 Received data:", data)  # ✅ Debugging log

        # ✅ Validate the required fields exist
        required_fields = ["temperature", "humidity", "oxygen_level", "light_illumination"]
        for field in required_fields:
            if field not in data:
                return Response({"error": f"Missing field: {field}"}, status=400)

        # ✅ Save to the database
        sensor_entry = SensorData.objects.create(
            temperature=data["temperature"],
            humidity=data["humidity"],
            oxygen_level=data["oxygen_level"],
            co2_level=data.get("co2_level", 400),  # ✅ Default CO2 = 400 if missing
            light_illumination=data["light_illumination"]
        )

        print("✅ Stored successfully in DB:", sensor_entry)

        return Response({"message": "Data stored successfully!"}, status=200)

    except Exception as e:
        print("❌ Error storing data:", str(e))  # ✅ Debugging log
        return Response({"error": str(e)}, status=500)  # ✅ Always return a response!

# ✅ Fetch AI Predictions (Last 24 Hours)
@api_view(['GET'])
def get_predictions(request):
    """Fetch AI-generated predictions from the database."""
    conn = sqlite3.connect("db.sqlite3")  # 🔥 Ensure correct DB path
    query = """
        SELECT timestamp, temperature, humidity, oxygen_level, co2_level, light_illumination
        FROM sensor_predictions
        ORDER BY timestamp DESC
        LIMIT 24
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    return JsonResponse(df.to_dict(orient="records"), safe=False)
