from typing import Any

from agent.realtime.realtime_api import OpenAIRealtimeAPIConfig
from agent.realtime.websocket_manager import WebSocketManager
from shared.logging_mixin import LoggingMixin


class RealtimeSessionMessageManager(LoggingMixin):
    """
    Manages OpenAI API session details and configuration.
    Separates session configuration from the main class.
    """

    def __init__(
        self, ws_manager: WebSocketManager, realtime_config: OpenAIRealtimeAPIConfig
    ):
        self.ws_manager = ws_manager
        self.system_message = realtime_config.system_message
        self.voice = realtime_config.voice
        self.temperature = realtime_config.temperature
        self.logger.info("SessionManager initialized")

    async def initialize_session(
        self,
    ) -> bool:
        """
        Initializes a session with the OpenAI API.
        """
        if not self.ws_manager.is_connected():
            self.logger.error("No connection available for session initialization")
            return False

        session_update = self._build_session_config()

        try:
            self.logger.info("Sending session update...")
            success = await self.ws_manager.send_message(session_update)

            if success:
                self.logger.info("Session update sent successfully")
                return True

            self.logger.error("Failed to send session update")
            return False

        except Exception as e:
            self.logger.error("Error initializing session: %s", e)
            return False

    def _build_session_config(self) -> dict[str, Any]:
        """
        Creates the session configuration for the OpenAI API.
        """
        return {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": self.voice,
                "instructions": self.system_message,
                "modalities": ["text", "audio"],
                "temperature": self.temperature,
                "tools": [],
            },
        }
