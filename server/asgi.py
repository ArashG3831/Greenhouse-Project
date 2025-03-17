import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
import sensor.routing  # ✅ Import WebSocket routes
from channels.auth import AuthMiddlewareStack

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),  # ✅ Handles HTTP requests
        "websocket": AuthMiddlewareStack(  # ✅ Handles WebSockets
            URLRouter(sensor.routing.websocket_urlpatterns)
        ),
    }
)
