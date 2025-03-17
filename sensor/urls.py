from django.urls import path
from .views import get_control_state, set_control_state, get_data, receive_data, get_predictions

urlpatterns = [
    path('api/get_data/', get_data, name='get_data'),
    path('api/receive_data/', receive_data, name='receive_data'),
    path('api/get_predictions/', get_predictions, name='get_predictions'),
    path('api/get_control_state/', get_control_state, name='get_control_state'),
    path('api/set_control_state/', set_control_state, name='set_control_state'),  # ✅ THIS WAS MISSING
]
