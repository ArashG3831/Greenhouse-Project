from django.urls import path
from .views import (
    get_control_state,
    set_control_state,
    get_data,
    receive_data,
    get_predictions,
    update_fan_status,
    smartthings_webhook  # ✅ Added import for new webhook
)

urlpatterns = [
    path('api/get_data/', get_data, name='get_data'),
    path('api/receive_data/', receive_data, name='receive_data'),
    path('api/get_predictions/', get_predictions, name='get_predictions'),
    path('api/get_control_state/', get_control_state, name='get_control_state'),
    path('api/set_control_state/', set_control_state, name='set_control_state'),
    path('api/smartthings_webhook/', smartthings_webhook, name='smartthings_webhook'),  # ✅ Added this new
    path('api/update-fan-status/', update_fan_status, name='update_fan_status'),
]