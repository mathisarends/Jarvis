from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
from typing import Any, Optional

from pydantic import BaseModel
import websocket
from dotenv import load_dotenv

from agent.realtime.websocket.realtime_event_dispatcher import RealtimeEventDispatcher
from agent.realtime.views import RealtimeModel
from agent.realtime.event_bus import EventBus
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin

load_dotenv()


class WebSocketManager(LoggingMixin):
    """
    Class for managing WebSocket connections using websocket-client.
    Handles the creation, management, and closing of WebSocket connections
    as well as sending and receiving messages with event bus integration.
    """

    DEFAULT_BASE_URL = "wss://api.openai.com/v1/realtime"
    NO_CONNECTION_ERROR_MSG = "No connection available. Call create_connection() first."

    def __init__(
        self,
        websocket_url: str,
        headers: dict[str, str],
    ):
        """
        Initialize the WebSocket Manager.
        """
        self.websocket_url = websocket_url
        self.headers = [f"{k}: {v}" for k, v in headers.items()] if headers else []
        self.ws: Optional[websocket.WebSocketApp] = None
        self._connected = False
        self._connection_event = threading.Event()
        self._running = False
        self.event_bus = EventBus()
        self.event_dispatcher = RealtimeEventDispatcher()

        self.event_bus.attach_loop(asyncio.get_running_loop())
        self.logger.info("WebSocketManager initialized")

    @classmethod
    def from_model(
        cls,
        *,
        model: RealtimeModel = RealtimeModel.GPT_REALTIME,
    ) -> WebSocketManager:
        """
        Create a manager for a given model (enum or raw string).
        """
        api_key = cls._get_api_key_from_env()
        ws_url = cls._get_websocket_url(model.value)
        headers = cls._get_auth_header(api_key)
        return cls(ws_url, headers)

    async def create_connection(self) -> bool:
        """
        Create a WebSocket connection.
        """
        try:
            self.logger.info("Establishing connection to %s...", self.websocket_url)

            self.ws = websocket.WebSocketApp(
                self.websocket_url,
                header=self.headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            # Start WebSocket in separate thread
            self._running = True
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()

            # Wait for connection to be established (with timeout)
            connected = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._connection_event.wait(timeout=10)
            )

            if connected and self._connected:
                return True
            else:
                self.logger.error("Failed to establish connection within timeout")
                return False

        except Exception as e:
            self.logger.error("Error creating connection: %s", e)
            return False

    async def send_message(self, message: dict[str, Any] | BaseModel) -> bool:
        """
        Send a JSON message through the WebSocket connection.
        """
        if not self._connected or not self.ws:
            self.logger.error(self.NO_CONNECTION_ERROR_MSG)
            return False

        if isinstance(message, BaseModel):
            payload = message.model_dump(exclude_none=True)
        else:
            payload = message

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._send_message_sync, payload
            )
            return result
        except Exception as e:
            self.logger.error("Error sending message: %s", e)
            return False

    async def close(self) -> None:
        """
        Close the WebSocket connection gracefully.
        """
        if not self.ws:
            return

        try:
            self.logger.info("Closing connection...")
            self._running = False
            self._connected = False

            await asyncio.get_event_loop().run_in_executor(None, self._close_sync)
            self.ws = None
            self.logger.info("Connection closed")
        except Exception as e:
            self.logger.error("Error closing connection: %s", e)

    def is_connected(self) -> bool:
        """
        Check if the WebSocket connection is established and open.
        """
        return self._connected and self.ws is not None

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        self.logger.info("Connection successfully established!")
        self._connected = True
        self._connection_event.set()

    def _on_message(self, ws, message: str) -> None:
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            message_type = data.get("type", "")
            self.logger.debug("Received message: %s", message_type)

            # Delegate event processing to RealtimeEventDispatcher
            self.event_dispatcher.dispatch_event(data)

        except json.JSONDecodeError as e:
            self.logger.warning("Received malformed JSON message: %s", e)
        except Exception as e:
            self.logger.error("Error processing message: %s", e)

    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        self.logger.error("WebSocket error: %s", error)
        self._connected = False
        # Publish error event
        self.event_bus.publish_sync(
            VoiceAssistantEvent.ERROR_OCCURRED, {"error": str(error)}
        )

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        self.logger.info("Connection closed: %s %s", close_status_code, close_msg)
        self._connected = False
        self._running = False

    def _send_message_sync(self, message: dict[str, Any]) -> bool:
        """Synchronously send message through WebSocket."""
        self.ws.send(json.dumps(message))
        return True

    def _close_sync(self) -> None:
        """Synchronously close WebSocket connection."""
        if self.ws:
            self.ws.close()

    @classmethod
    def _get_websocket_url(cls, model: str) -> str:
        """Build the Realtime WebSocket URL with model as query parameter."""
        return f"{cls.DEFAULT_BASE_URL}?model={model}"

    @classmethod
    def _get_auth_header(
        cls,
        api_key: str,
    ) -> dict[str, str]:
        """Build Authorization and optional organization/project headers."""
        return {"Authorization": f"Bearer {api_key}"}

    @classmethod
    def _get_api_key_from_env(cls) -> str:
        """
        Load API key from OPENAI_API_KEY environment variable.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set OPENAI_API_KEY or provide api_key parameter."
            )
        return api_key
