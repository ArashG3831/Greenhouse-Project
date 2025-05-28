from django.utils.timezone import now, timedelta
from django.db.models import Avg, Min
from django.db.models.expressions import RawSQL
from django.utils import timezone
from .models import SensorData, SensorPrediction, ControlState
from django.views.decorators.cache import cache_page
from django.utils.timezone import localtime
from django.utils.timezone import now
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from .models import SensorData
# ----------------------- Sensor Data API -----------------------

@api_view(['GET'])
@cache_page(10)  # cache for 10 seconds
def get_data(request):
    try:
        time_range = request.GET.get('range', '7d')
        now_time = now()

        range_map = {
            '1h':  (now_time - timedelta(hours=1), 1),
            '24h': (now_time - timedelta(hours=24), 5),
            '7d':  (now_time - timedelta(days=7), 30),
            '30d': (now_time - timedelta(days=30), 60),
        }

        start_date, group_minutes = range_map.get(time_range, range_map['7d'])

        # Basic aggregation (without touching leaf_color)
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
                soil_moisture=Avg("soil_moisture"),
                co2_level=Avg("co2_level"),
                light_illumination=Avg("light_illumination"),
                leaf_color=Min("leaf_color"),
            )
            .order_by("minute_bucket")
        )

        data = list(queryset)

        return Response({"data": data})

    except Exception as e:
        print("❌ Error in get_data:", str(e))
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def get_latest(request):
    try:
        latest = SensorData.objects.order_by("-timestamp").first()
        if not latest:
            return Response({"error": "No sensor data available."}, status=404)

        return Response({
            "timestamp": localtime(latest.timestamp).isoformat(),
            "temperature": latest.temperature,
            "humidity": latest.humidity,
            "soil_moisture": latest.soil_moisture,
            "co2_level": latest.co2_level,
            "light_illumination": latest.light_illumination,
            "leaf_color": latest.leaf_color or "#00ff00",
        })


    except Exception as e:
        print("❌ Error in get_latest:", str(e))
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
@csrf_exempt
def receive_data(request):
    try:
        data = request.data if hasattr(request, "data") else request.POST
        print("🔥 Received data:", data)

        required_fields = [
            "temperature", "humidity", "light_illumination",
            "soil_moisture", "co2_level", "leaf_color"
        ]

        for field in required_fields:
            if field not in data:
                return Response({"error": f"Missing field: {field}"}, status=400)

        entry = SensorData.objects.create(
            temperature=float(data["temperature"]),
            humidity=float(data["humidity"]),
            light_illumination=float(data["light_illumination"]),
            soil_moisture=float(data["soil_moisture"]),
            co2_level=float(data["co2_level"]),
            leaf_color=data["leaf_color"]
        )

        print("✅ Stored successfully in DB:", entry)
        return Response({"message": "Data stored successfully!"})

    except Exception as e:
        print("❌ Error in receive_data:", str(e))
        return Response({"error": str(e)}, status=500)

# ----------------------- Prediction API -----------------------

@api_view(['GET'])
@cache_page(60)  # cache for 60 seconds
def get_predictions(request):
    try:
        predictions = (
            SensorPrediction.objects
            .order_by("-timestamp")[:24]
            .values("timestamp", "temperature", "humidity", "soil_moisture", "co2_level", "light_illumination")
        )
        return Response(list(reversed(predictions)))

    except Exception as e:
        print("❌ Error in get_predictions:", str(e))
        return Response({"error": str(e)}, status=500)


# ----------------------- Control State API -----------------------

@api_view(['GET'])
# @cache_page(1)  # Cache for just 1 second
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
