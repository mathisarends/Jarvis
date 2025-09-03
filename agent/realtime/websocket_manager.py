from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
from typing import Any, Callable, Dict, Optional

import websocket
from dotenv import load_dotenv

from agent.realtime.views import RealtimeModel
from shared.logging_mixin import LoggingMixin

load_dotenv()


class WebSocketManager(LoggingMixin):
    """
    Class for managing WebSocket connections using websocket-client.
    Handles the creation, management, and closing of WebSocket connections
    as well as sending and receiving messages.
    """

    DEFAULT_BASE_URL = "wss://api.openai.com/v1/realtime"
    NO_CONNECTION_ERROR_MSG = "No connection available. Call create_connection() first."

    def __init__(
        self,
        websocket_url: str,
        headers: Dict[str, str],
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
        self.logger.info("WebSocketManager initialized")


    @classmethod
    def for_gpt_realtime(
        cls, *, api_key: str | None = None
    ) -> WebSocketManager:
        """
        Convenience factory for 'gpt-realtime'.
        """
        return cls._from_model(
            api_key=api_key,
            model=RealtimeModel.GPT_REALTIME,
        )

    async def create_connection(self) -> bool:
        """
        Create a WebSocket connection.
        """
        try:
            self.logger.info("Establishing connection to %s...", self.websocket_url)
            
            def on_open(ws):
                self.logger.info("Connection successfully established!")
                self._connected = True
                self._connection_event.set()

            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self.logger.debug("Received message: %s", data.get("type"))
                    # Handle message directly here - user can implement custom logic
                    # For now, just log the message type
                    
                except json.JSONDecodeError as e:
                    self.logger.warning("Received malformed JSON message: %s", e)
                except Exception as e:
                    self.logger.error("Error processing message: %s", e)

            def on_error(ws, error):
                self.logger.error("WebSocket error: %s", error)
                self._connected = False

            def on_close(ws, close_status_code, close_msg):
                self.logger.info("Connection closed: %s %s", close_status_code, close_msg)
                self._connected = False
                self._running = False

            self.ws = websocket.WebSocketApp(
                self.websocket_url,
                header=self.headers,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
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

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a JSON message through the WebSocket connection.
        """
        if not self._connected or not self.ws:
            self.logger.error(self.NO_CONNECTION_ERROR_MSG)
            return False

        try:
            def _send():
                self.ws.send(json.dumps(message))
                return True
            
            result = await asyncio.get_event_loop().run_in_executor(None, _send)
            return result
        except Exception as e:
            self.logger.error("Error sending message: %s", e)
            return False

    async def send_binary(self, data: bytes, encoding: str = "base64") -> bool:
        """
        Send binary data through the WebSocket connection.
        For audio streaming, data is typically encoded in base64.
        """
        if not self._connected or not self.ws:
            self.logger.error(self.NO_CONNECTION_ERROR_MSG)
            return False

        if encoding != "base64":
            self.logger.error("Unsupported encoding: %s", encoding)
            return False

        try:
            base64_data = base64.b64encode(data).decode("utf-8")
            message = {
                "type": "input_audio_buffer.append",
                "audio": base64_data,
            }
            return await self.send_message(message)
        except Exception as e:
            self.logger.error("Error sending binary data: %s", e)
            return False

    async def receive_messages(
        self,
        should_continue: Callable[[], bool] = lambda: True,
    ) -> None:
        """
        Monitor the WebSocket connection status.
        Messages are now handled directly in the on_message callback.
        """
        if not self._connected:
            self.logger.error(self.NO_CONNECTION_ERROR_MSG)
            return

        try:
            self.logger.info("Monitoring WebSocket connection...")
            
            while should_continue() and self._running and self._connected:
                # Messages are handled directly in on_message callback
                # Just keep the connection alive
                await asyncio.sleep(0.1)
                        
            self.logger.info("Connection monitoring stopped")

        except Exception as e:
            self.logger.error("Error monitoring connection: %s", e)

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
            
            def _close():
                if self.ws:
                    self.ws.close()
            
            await asyncio.get_event_loop().run_in_executor(None, _close)
            self.ws = None
            self.logger.info("Connection closed")
        except Exception as e:
            self.logger.error("Error closing connection: %s", e)

    def is_connected(self) -> bool:
        """
        Check if the WebSocket connection is established and open.
        """
        return self._connected and self.ws is not None

    async def _process_message(self, message: Dict[str, Any]) -> None:
        """
        This method is no longer used since messages are handled directly
        in the on_message callback. Kept for potential future use.
        """
        # Messages are now handled directly in on_message callback
        pass

    @classmethod
    def _from_model(
        cls,
        *,
        api_key: str | None = None,
        model: RealtimeModel = RealtimeModel.GPT_REALTIME,
    ) -> WebSocketManager:
        """
        Create a manager for a given model (enum or raw string).
        """
        actual_api_key = api_key or cls._get_api_key_from_env()
        ws_url = cls._get_websocket_url(model.value)
        headers = cls._get_auth_header(actual_api_key)
        return cls(ws_url, headers)

    @classmethod
    def _get_websocket_url(cls, model: str) -> str:
        """Build the Realtime WebSocket URL with model as query parameter."""
        return f"{cls.DEFAULT_BASE_URL}?model={model}"

    @classmethod
    def _get_auth_header(
        cls,
        api_key: str,
    ) -> Dict[str, str]:
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