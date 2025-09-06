from agent.config.views import AgentConfig
from agent.realtime.events.client.session_update import (
    AudioFormat,
    AudioFormatConfig,
    AudioInputConfig,
    AudioOutputConfig,
    RealtimeSessionConfig,
    SessionUpdateEvent,
    AudioConfig,
)
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.websocket.websocket_manager import WebSocketManager
from shared.logging_mixin import LoggingMixin


class SessionManager(LoggingMixin):
    """Handles session configuration and initialization."""

    def __init__(
        self,
        agent_config: AgentConfig,
        tool_registry: ToolRegistry,
        ws_manager: WebSocketManager,
    ):
        self.agent_config = agent_config
        self.tool_registry = tool_registry
        self.ws_manager = ws_manager

    async def initialize(self) -> bool:
        """Initialize session with OpenAI API."""
        if not self.ws_manager.is_connected():
            self.logger.error("No connection available for session initialization")
            return False

        try:
            session_config = self._build_config()
            self.logger.info("Sending session update...")

            success = await self.ws_manager.send_message(session_config)
            if success:
                self.logger.info("Session update sent successfully")
            else:
                self.logger.error("Failed to send session update")

            return success
        except Exception as e:
            self.logger.error("Error initializing session: %s", e)
            return False

    def _build_config(self) -> SessionUpdateEvent:
        """Build session configuration."""
        audio_config = AudioConfig(
            output=AudioOutputConfig(
                format=AudioFormatConfig(type=AudioFormat.PCM16),
                speed=self.agent_config.speed,
                voice=self.agent_config.voice,
            ),
            input=AudioInputConfig(
                transcription=self.agent_config.transcription,
                noise_reduction=self.agent_config.input_audio_noise_reduction,
            ),
        )

        import json

        test = self.tool_registry.get_openai_schema()
        print(json.dumps([tool.model_dump() for tool in test], indent=2))

        return SessionUpdateEvent(
            session=RealtimeSessionConfig(
                type="realtime",
                model=self.agent_config.model,
                instructions=self.agent_config.instructions,
                audio=audio_config,
                output_modalities=["audio"],
                max_output_tokens=self.agent_config.max_response_output_tokens,
                tools=self.tool_registry.get_openai_schema(),
            ),
        )
