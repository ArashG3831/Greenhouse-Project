from django.utils.timezone import now, timedelta
from django.db.models import Avg, F, Min, ExpressionWrapper, IntegerField
from django.db.models.expressions import RawSQL
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.utils import timezone
from .models import SensorData, SensorPrediction, ControlState
from django.utils.timezone import now
from zoneinfo import ZoneInfo
from django.db.models import Max

@api_view(['GET'])
def get_data(request):
    time_range = request.GET.get('range', '7d')

    now_time = now()

    # Define the start time and downsampling interval (in minutes)
    ranges = {
        '1h': (now_time - timedelta(hours=1), 1),  # No downsampling
        '24h': (now_time - timedelta(hours=24), 5),  # Group every 5 minutes
        '7d': (now_time - timedelta(days=7), 30),  # Group every 30 minutes
        '30d': (now_time - timedelta(days=30), 60),  # Group hourly
    }

    start_date, group_minutes = ranges.get(time_range, (now_time - timedelta(days=7), 30))

    # Use time-flooring SQL-level truncation for grouping
    queryset = (
        SensorData.objects
        .filter(timestamp__gte=start_date)
        .annotate(minute_bucket=RawSQL(
            "FROM_UNIXTIME(FLOOR(UNIX_TIMESTAMP(timestamp) / %s) * %s)",
            (group_minutes * 60, group_minutes * 60)
        ))
        .values("minute_bucket")
        .annotate(
            timestamp=Min("timestamp"),
            temperature=Avg("temperature"),
            humidity=Avg("humidity"),
            oxygen_level=Avg("oxygen_level"),
            co2_level=Avg("co2_level"),
            light_illumination=Avg("light_illumination"),
        )
        .order_by("minute_bucket")
    )

    # Get latest true timestamp for front-end "last updated"
    latest_ts = (
        SensorData.objects
        .latest("timestamp")
        .timestamp
    )

    return Response({
        "data": list(queryset),
        "latest_timestamp": latest_ts
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

        entry = SensorData.objects.create(
            timestamp=now(),
            temperature=data["temperature"],
            humidity=data["humidity"],
            oxygen_level=data["oxygen_level"],
            co2_level=data.get("co2_level", 400),
            light_illumination=data["light_illumination"]
        )

        print("✅ Stored successfully in DB:", entry)
        return Response({"message": "Data stored successfully!"})

    except Exception as e:
        print("❌ Error:", str(e))
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def get_predictions(request):
    try:
        # Fetch the latest 24 prediction entries
        predictions = (
            SensorPrediction.objects.order_by("-timestamp")[:24]
            .values("timestamp", "temperature", "humidity", "oxygen_level", "co2_level", "light_illumination")
        )

        # Return as a list (in chronological order)
        return Response(list(reversed(predictions)))

    except Exception as e:
        print("❌ Error in get_predictions:", str(e))
        return Response({"error": str(e)}, status=500)


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
def set_control_state(request):
    control, _ = ControlState.objects.get_or_create(id=1)

    fan_mode = request.data.get("fan_mode")
    if fan_mode in ["auto", "on", "off"]:
        control.fan_mode = fan_mode
        control.fan_is_running = fan_mode in ["on", "auto"]

    water_mode = request.data.get("water_mode")
    if water_mode in ["auto", "off"]:
        control.water_mode = water_mode
    elif water_mode == "+10ml":
        control.last_water_dispense = timezone.now()

    control.save()
    return Response({
        "message": "Control state updated!",
        "fan_mode": control.fan_mode,
        "fan_is_running": control.fan_is_running,
        "water_mode": control.water_mode,
        "last_water_dispense": control.last_water_dispense
    })


# @api_view(['POST'])
# def update_fan_status(request):
#     device_id = request.data.get("deviceId")
#     state = request.data.get("state")
#
#     if not device_id or not state:
#         return Response({"error": "Missing deviceId or state"}, status=400)
#
#     if device_id != "9975afad-dea0-477e-a5a3-6586d8da3f8a":
#         return Response({"error": "Invalid device ID"}, status=400)
#
#     if state not in ["on", "off", "auto", "Low", "Medium", "High"]:
#         return Response({"error": "Invalid state value"}, status=400)
#
#     control, _ = ControlState.objects.get_or_create(id=1)
#     control.fan_mode = state
#     control.fan_is_running = state in ["on", "auto", "Low", "Medium", "High"]
#     control.save()
#
#     print(f"Webhook update: device {device_id} set to {state}")
#     return Response({"message": "Fan status updated!", "fan_mode": control.fan_mode})
#
#
# @api_view(['POST'])
# def smartthings_webhook(request):
#     data = request.data
#     lifecycle = data.get('lifecycle')
#
#     if lifecycle == 'INSTALL':
#         pass  # Stub for install hook
#
#     if lifecycle == 'EXECUTE':
#         commands = data.get('executeData', {}).get('commands', [])
#         control, _ = ControlState.objects.get_or_create(id=1)
#
#         for cmd in commands:
#             capability = cmd.get('capability')
#             command = cmd.get('command')
#             args = cmd.get('arguments', [])
#
#             if capability == 'switch':
#                 if command == 'on':
#                     control.fan_mode = 'on'
#                     control.fan_is_running = True
#                 elif command == 'off':
#                     control.fan_mode = 'off'
#                     control.fan_is_running = False
#
#             elif capability == 'fanMode':
#                 if command == 'setFanMode' and args:
#                     mode = args[0]
#                     control.fan_mode = mode
#                     control.fan_is_running = (mode.lower() != 'off')
#
#         control.save()
#         return Response({"status": "success", "fan_mode": control.fan_mode})
#
#     return Response({"status": "ignored"})
