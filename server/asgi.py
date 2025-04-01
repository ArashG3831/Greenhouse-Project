import os
import django
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import sensor.routing  # WebSocket routes

# Ensure Django settings are loaded before anything else
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
django.setup()

# ASGI application with support for both HTTP and WebSocket
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(sensor.routing.websocket_urlpatterns)
    ),
})
