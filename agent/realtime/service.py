import os, json
import websocket

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
headers = ["Authorization: Bearer " + OPENAI_API_KEY]


def on_open(ws):
    print("✅ Connected to Realtime API")


def on_message(ws, message):
    event = json.loads(message)
    print("📥 Received event:", event)


ws = websocket.WebSocketApp(
    url,
    header=headers,
    on_open=on_open,
    on_message=on_message,
)

ws.run_forever()
