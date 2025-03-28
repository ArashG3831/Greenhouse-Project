from django.http import JsonResponse
import sqlite3
import pandas as pd
from django.db.models import Avg, F, Min, ExpressionWrapper, IntegerField
from django.utils.timezone import now, timedelta
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
from .models import SensorData, ControlState  # Imported ControlState!

@api_view(['GET'])
def get_data(request):
    time_range = request.GET.get('range', '7d')  # Default to Last 7 Days

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
    control, _ = ControlState.objects.get_or_create(id=1)

    # Update fan mode if provided
    fan_mode = request.data.get("fan_mode")
    if fan_mode in ["auto", "on", "off"]:
        control.fan_mode = fan_mode
        if fan_mode in ["on", "auto"]:  # When auto or on, set fan as running.
            control.fan_is_running = True
        elif fan_mode == "off":
            control.fan_is_running = False

    # Handle water control update
    water_mode = request.data.get("water_mode")
    if water_mode in ["auto", "off"]:
        control.water_mode = water_mode
    elif water_mode == "+10ml":
        control.last_water_dispense = timezone.now()
        # Optionally reset water_mode if desired:
        # control.water_mode = "off"

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
    control, _ = ControlState.objects.get_or_create(id=1)
    return Response({
        "fan_mode": control.fan_mode,
        "fan_is_running": control.fan_is_running,
        "water_mode": control.water_mode,
        "last_water_dispense": control.last_water_dispense
    })

@api_view(['POST'])
def smartthings_webhook(request):
    data = request.data
    lifecycle = data.get('lifecycle')

    # 1) Handle CONFIRMATION
    if lifecycle == 'CONFIRMATION':
        confirmation_url = data['confirmationData']['confirmationUrl']
        return Response({
            'statusCode': 200,
            'confirmationData': {
                'confirmationUrl': confirmation_url
            }
        })

    # 2) Handle PING
    if lifecycle == 'PING':
        return Response({
            'statusCode': 200,
            'pingData': {
                'challenge': data['pingData']['challenge']
            }
        })

    # 3) Handle INSTALL/UPDATE/UNINSTALL if needed
    if lifecycle == 'INSTALL':
        pass
    elif lifecycle == 'UPDATE':
        pass
    elif lifecycle == 'UNINSTALL':
        pass

    # 4) Handle EXECUTE or commands from your existing logic
    commands = data.get("commands", [])
    if not commands:
        return Response({"error": "No commands received"}, status=400)

    control, _ = ControlState.objects.get_or_create(id=1)
    for command in commands:
        capability = command.get('capability')
        action = command.get('command')
        if capability == "switch":
            if action == "on":
                control.fan_mode = "on"
                control.fan_is_running = True
            elif action == "off":
                control.fan_mode = "off"
                control.fan_is_running = False
        elif capability == "momentary":
            if action == "push":
                control.last_water_dispense = timezone.now()
    control.save()
    return Response({"status": "success", "fan_mode": control.fan_mode})

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
    try:
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
    except Exception as e:
        print("Error in get_predictions:", str(e))
        return Response({"error": str(e)}, status=500)

# --- New Endpoint: update_fan_status ---
@api_view(['POST'])
def update_fan_status(request):
    data = request.data
    device_id = data.get("deviceId")
    state = data.get("state")

    if not device_id or not state:
        return Response({"error": "Missing deviceId or state"}, status=400)

    expected_device_id = "9975afad-dea0-477e-a5a3-6586d8da3f8a"
    if device_id != expected_device_id:
        return Response({"error": "Invalid device ID"}, status=400)

    # Update the allowed state values:
    allowed_states = ["on", "off", "auto", "Low", "Medium", "High"]
    if state not in allowed_states:
        return Response({"error": "Invalid state value"}, status=400)

    control, _ = ControlState.objects.get_or_create(id=1)
    control.fan_mode = state
    control.fan_is_running = state in ["on", "auto", "Low", "Medium", "High"]  # Adjust logic as needed
    control.save()

    print(f"Webhook update: device {device_id} set to {state}")
    return Response({"message": "Fan status updated!", "fan_mode": control.fan_mode})
