from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import SensorData, ControlState
import sqlite3
import pandas as pd
from django.db.models import Avg, F, Min, ExpressionWrapper, IntegerField
from django.utils.timezone import now, timedelta

@api_view(['GET'])
def get_data(request):
    # (Same as before; unchanged)
    time_range = request.GET.get('range', '7d')
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
    else:
        start_date = None
        group_factor = 2880

    if start_date:
        data = list(
            SensorData.objects.filter(timestamp__gte=start_date)
            .annotate(group_id=ExpressionWrapper(F("id") / group_factor, output_field=IntegerField()))
            .values("group_id")
            .annotate(
                timestamp=Min("timestamp"),
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
            .annotate(group_id=ExpressionWrapper(F("id") / group_factor, output_field=IntegerField()))
            .values("group_id")
            .annotate(
                timestamp=Min("timestamp"),
                temperature=Avg("temperature"),
                humidity=Avg("humidity"),
                oxygen_level=Avg("oxygen_level"),
                co2_level=Avg("co2_level"),
                light_illumination=Avg("light_illumination"),
            )
            .order_by("timestamp")
        )

    latest_entry = (
        SensorData.objects.order_by("-timestamp")
        .values("timestamp", "temperature", "humidity", "oxygen_level", "co2_level", "light_illumination")
        .first()
    )
    if latest_entry and (not data or latest_entry["timestamp"] != data[-1]["timestamp"]):
        data.append(latest_entry)

    return Response(data)

@api_view(['POST'])
def set_control_state(request):
    """Updates the control mode for Fan & Water and handles a one-time water dispense command."""
    control, _ = ControlState.objects.get_or_create(id=1)

    # Update fan mode if provided
    fan_mode = request.data.get("fan_mode")
    if fan_mode in ["auto", "on", "off"]:
        control.fan_mode = fan_mode
        # Update runtime fan status: if set to "on", we consider it running.
        if fan_mode == "on":
            control.fan_is_running = True
        elif fan_mode == "off":
            control.fan_is_running = False
        # In auto mode, you might update control.fan_is_running based on sensor logic

    # Handle water control update
    water_mode = request.data.get("water_mode")
    if water_mode in ["auto", "off"]:
        control.water_mode = water_mode
    elif water_mode == "+10ml":
        # Update last water dispense time
        control.last_water_dispense = timezone.now()
        # Optionally, you could change water_mode here if needed

    control.save()

    return Response({
        "message": "Control state updated!",
        "fan_mode": control.fan_mode,
        "fan_is_running": control.fan_is_running,
        "water_mode": control.water_mode,
        "last_water_dispense": control.last_water_dispense
    })

@api_view(['GET'])
def get_control_state(request):
    """Returns the current control state including fan running status and last water dispense time."""
    control, _ = ControlState.objects.get_or_create(id=1)
    return Response({
        "fan_mode": control.fan_mode,
        "fan_is_running": control.fan_is_running,
        "water_mode": control.water_mode,
        "last_water_dispense": control.last_water_dispense
    })

@api_view(['POST'])
def receive_data(request):
    try:
        data = request.data
        print("🔥 Received data:", data)
        required_fields = ["temperature", "humidity", "oxygen_level", "light_illumination"]
        for field in required_fields:
            if field not in data:
                return Response({"error": f"Missing field: {field}"}, status=400)
        sensor_entry = SensorData.objects.create(
            temperature=data["temperature"],
            humidity=data["humidity"],
            oxygen_level=data["oxygen_level"],
            co2_level=data.get("co2_level", 400),
            light_illumination=data["light_illumination"]
        )
        print("✅ Stored successfully in DB:", sensor_entry)
        return Response({"message": "Data stored successfully!"}, status=200)
    except Exception as e:
        print("❌ Error storing data:", str(e))
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def get_predictions(request):
    conn = sqlite3.connect("db.sqlite3")
    query = """
        SELECT timestamp, temperature, humidity, oxygen_level, co2_level, light_illumination
        FROM sensor_predictions
        ORDER BY timestamp DESC
        LIMIT 24
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return JsonResponse(df.to_dict(orient="records"), safe=False)
