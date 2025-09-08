from agent.config.views import AgentConfig
from agent.realtime.event_bus import EventBus
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
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


class SessionManager(LoggingMixin):
    """Handles session configuration and initialization."""

    def __init__(
        self,
        agent_config: AgentConfig,
        tool_registry: ToolRegistry,
        ws_manager: WebSocketManager,
        event_bus: EventBus,
    ):
        self.agent_config = agent_config
        self.tool_registry = tool_registry
        self.ws_manager = ws_manager
        self.event_bus = event_bus
        
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_CONFIG_UPDATE_REQUEST,
            self._handle_config_update_request
        )

    async def _send_session_update(self):
        """Send updated session configuration to OpenAI."""
        try:
            session_config = self._build_config()
            self.logger.info("Sending session update with new config...")

            success = await self.ws_manager.send_message(session_config)
            if success:
                self.logger.info("Session update sent successfully")
            else:
                self.logger.error("Failed to send session update")
        except Exception as e:
            self.logger.error("Error sending session update: %s", e)

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

        return SessionUpdateEvent(
            session=RealtimeSessionConfig(
                model=self.agent_config.model,
                instructions=self.agent_config.instructions,
                audio=audio_config,
                output_modalities=["audio"],
                max_output_tokens=self.agent_config.max_response_output_tokens,
                tools=self.tool_registry.get_openai_schema(),
            ),
        )


    async def _handle_config_update_request(self, event: VoiceAssistantEvent, new_response_speed: float):
        """Handle assistant configuration update requests."""
        self.logger.info(
            "Received config update request - New response speed: %.2f",
            new_response_speed
        )

        # Update the agent config with new response speed
        self.agent_config.speed = new_response_speed

        # Send updated session configuration to OpenAI
        await self._send_session_update()