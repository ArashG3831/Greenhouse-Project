import json
from channels.generic.websocket import AsyncWebsocketConsumer

class SensorConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("✅ WebSocket Connected!")  # Debugging

    async def disconnect(self, close_code):
        print("❌ WebSocket Disconnected!")

    async def receive(self, text_data):
        data = json.loads(text_data)
        print("🔄 Received WebSocket Data:", data)  # Debugging

        await self.send(text_data=json.dumps({"message": "Received!"}))
