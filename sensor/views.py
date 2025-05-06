from django.utils.timezone import now, timedelta
from django.db.models import Avg, F, Min, ExpressionWrapper, IntegerField
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse
from django.utils import timezone
from .models import SensorData, ControlState


@api_view(['GET'])
def get_data(request):
    time_range = request.GET.get('range', '7d')
    group_factor = 720
    start_date = None

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

    queryset = SensorData.objects.all()
    if start_date:
        queryset = queryset.filter(timestamp__gte=start_date)

    grouped_data = list(
        queryset
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

    latest = (
        SensorData.objects.order_by("-timestamp")
        .values("timestamp", "temperature", "humidity", "oxygen_level", "co2_level", "light_illumination")
        .first()
    )

    if latest and (not grouped_data or latest["timestamp"] > grouped_data[-1]["timestamp"]):
        grouped_data.append(latest)

    return Response(grouped_data)


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
