from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, Callable, Dict, Optional

import websockets
from websockets.asyncio.client import connect, ClientConnection
from dotenv import load_dotenv

from agent.realtime.views import RealtimeModel
from shared.logging_mixin import LoggingMixin

load_dotenv()


class WebSocketManager(LoggingMixin):
    """
    Class for managing WebSocket connections.
    Handles the creation, management, and closing of WebSocket connections
    as well as sending and receiving messages.
    """

    DEFAULT_BASE_URL = "wss://api.openai.com/v1/realtime"
    NO_CONNECTION_ERROR_MSG = "No connection available. Call create_connection() first."

    def __init__(self, websocket_url: str, headers: Dict[str, str], event_router=None):
        """
        Initialize the WebSocket Manager.
        """
        self.websocket_url = websocket_url
        self.headers = list(headers.items()) if headers else None
        self.connection: Optional[ClientConnection] = None
        self.event_router = event_router
        self.logger.info("WebSocketManager initialized")

    @classmethod
    def for_gpt_realtime(
        cls, *, api_key: str | None = None, event_router=None
    ) -> WebSocketManager:
        """
        Convenience factory for 'gpt-realtime'.
        """
        return cls._from_model(
            api_key=api_key, model=RealtimeModel.GPT_REALTIME, event_router=event_router
        )

    async def create_connection(self) -> Optional[ClientConnection]:
        """
        Create a WebSocket connection.
        """
        try:
            self.logger.info("Establishing connection to %s...", self.websocket_url)
            self.connection = await connect(
                self.websocket_url, extra_headers=self.headers
            )
            self.logger.info("Connection successfully established!")
            return self.connection

        except websockets.exceptions.InvalidStatus as e:
            self.logger.error("Invalid status code from WebSocket server: %s", e)
        except ConnectionRefusedError as e:
            self.logger.error("Connection refused: %s", e)
        except OSError as e:
            self.logger.error("OS-level connection error: %s", e)
        return None

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a JSON message through the WebSocket connection.
        """
        if not self.connection:
            self.logger.error(self.NO_CONNECTION_ERROR_MSG)
            return False

        try:
            await self.connection.send(json.dumps(message))
            return True
        except Exception as e:
            self.logger.error("Error sending message: %s", e)
            return False

    async def send_binary(self, data: bytes, encoding: str = "base64") -> bool:
        """
        Send binary data through the WebSocket connection.
        For audio streaming, data is typically encoded in base64.
        """
        if not self.connection:
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
        Continuously receive and process messages from the WebSocket connection.
        """
        if not self.connection:
            self.logger.error(self.NO_CONNECTION_ERROR_MSG)
            return

        try:
            self.logger.info("Starting message reception...")
            async for message in self.connection:
                await self._process_websocket_message(message)

                if not should_continue():
                    break

        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info(
                "WebSocket connection closed normally during message reception"
            )
        except websockets.exceptions.ConnectionClosedError as e:
            if str(e).startswith("sent 1000 (OK)"):
                self.logger.info(
                    "WebSocket connection closed normally during message reception"
                )
            else:
                self.logger.error(
                    "WebSocket connection closed unexpectedly during message reception: %s",
                    e,
                )
        except asyncio.TimeoutError as e:
            self.logger.error("Timeout while receiving messages: %s", e)

    async def close(self) -> None:
        """
        Close the WebSocket connection gracefully.
        """
        if not self.connection:
            return

        try:
            self.logger.info("Closing connection...")
            await self.connection.close()
            self.connection = None
            self.logger.info("Connection closed")
        except Exception as e:
            self.logger.error("Error closing connection: %s", e)

    def is_connected(self) -> bool:
        """
        Check if the WebSocket connection is established and open.
        """
        return self.connection is not None and self.connection.open

    async def _process_websocket_message(self, message: str) -> None:
        """
        Process incoming WebSocket messages and route events.

        Args:
            message: The raw message from the WebSocket
        """
        try:
            self.logger.debug("Raw message received: %s...", message[:100])

            response = json.loads(message)

            if not isinstance(response, dict):
                self.logger.warning("Response is not a dictionary: %s", type(response))
                return

            event_type = response.get("type", "")
            await self.event_router.process_event(event_type, response)

        except json.JSONDecodeError as e:
            self.logger.warning("Received malformed JSON message: %s", e)
        except KeyError as e:
            self.logger.warning(
                "Expected key missing in message: %s | Message content: %s",
                e,
                message[:500],
            )
        except Exception as e:
            self.logger.error("Unexpected error processing message: %s", e)

    @classmethod
    def _from_model(
        cls,
        *,
        api_key: str | None = None,
        model: RealtimeModel = RealtimeModel.GPT_REALTIME,
        event_router=None,
    ) -> WebSocketManager:
        """
        Create a manager for a given model (enum or raw string).
        """
        actual_api_key = api_key or cls._get_api_key_from_env()
        ws_url = cls._get_websocket_url(model.value)
        headers = cls._get_auth_header(actual_api_key)
        return cls(ws_url, headers, event_router)

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
