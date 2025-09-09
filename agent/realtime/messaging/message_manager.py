from agent.config.views import AgentConfig
from agent.realtime.current_message_context import CurrentMessageContext
from agent.realtime.event_bus import EventBus
from agent.realtime.events.client.conversation_item_truncate import (
    ConversationItemTruncateEvent,
)
from agent.realtime.events.client.conversation_response_create import (
    ConversationResponseCreateEvent,
)
from agent.realtime.events.client.session_update import (
    AudioFormat,
    AudioFormatConfig,
    AudioInputConfig,
    AudioOutputConfig,
    RealtimeSessionConfig,
    SessionUpdateEvent,
    AudioConfig,
)
from agent.realtime.messaging.message_queue import MessageQueue
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.views import FunctionCallResult
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


class RealtimeMessageManager(LoggingMixin):
    """
    Clean orchestrator for realtime message management.
    Handles all message operations directly without separate handler classes.
    """

    def __init__(
        self,
        ws_manager: WebSocketManager,
        tool_registry: ToolRegistry,
        agent_config: AgentConfig,
        event_bus: EventBus,
    ):
        self.ws_manager = ws_manager
        self.event_bus = event_bus
        self.agent_config = agent_config
        self.tool_registry = tool_registry
        self.current_message_context = CurrentMessageContext(self.event_bus)

        # Initialize specialized components
        self.queue = MessageQueue()

        # Setup event handling
        self.event_bus = event_bus
        self._setup_event_handlers()

    async def initialize_session(self) -> bool:
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

    async def send_tool_result(self, function_call_result: FunctionCallResult) -> None:
        """Send tool result (queued if response active)."""
        await self.queue.send_or_queue(
            self._send_tool_result_direct, function_call_result
        )

    async def send_execution_message(self, message: str) -> None:
        """Send update message for generator tool progress (queued if response active)."""
        await self.queue.send_or_queue(self._send_execution_message_direct, message)

    async def _send_tool_result_direct(
        self, function_call_result: FunctionCallResult
    ) -> None:
        """Send tool result to OpenAI Realtime API."""
        try:
            self.logger.info(
                "Sending tool result for '%s'", function_call_result.tool_name
            )

            # Send conversation item
            if not await self._send_conversation_item(function_call_result):
                return

            # Trigger response
            await self._trigger_response(function_call_result)

        except Exception as e:
            self.logger.error(
                "Error handling tool result for '%s': %s",
                function_call_result.tool_name,
                e,
                exc_info=True,
            )

    async def _send_execution_message_direct(self, message: str) -> None:
        """
        Send a generator tool progress update as a conversation response.
        """
        try:
            self.logger.info("Sending generator tool update")

            response_event = ConversationResponseCreateEvent.with_instructions(
                f"Say exactly: '{message}'. Do not add any information not in this message."
            )

            if await self.ws_manager.send_message(response_event):
                self.logger.info("Generator tool update sent successfully")
            else:
                self.logger.error("Failed to send response.create for generator update")

        except Exception as e:
            self.logger.error("Error sending generator update: %s", e, exc_info=True)

    async def _send_conversation_item(self, result: FunctionCallResult) -> bool:
        """Send function call output as conversation item."""
        conversation_item = result.to_conversation_item()
        success = await self.ws_manager.send_message(conversation_item)

        if not success:
            self.logger.error(
                "Failed to send function_call_output for '%s'", result.tool_name
            )

        return success

    async def _trigger_response(self, result: FunctionCallResult) -> None:
        """Trigger response creation."""
        response_event = ConversationResponseCreateEvent.with_instructions(
            result.response_instruction
            or "Process the tool result and provide a helpful response."
        )
        await self.ws_manager.send_message(response_event)

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
                tools=self.tool_registry.get_openai_schema(),
            ),
        )

    async def _handle_config_update_request(self, new_response_speed: float):
        """Handle assistant configuration update requests."""
        self.logger.info(
            "Received config update request - New response speed: %.2f",
            new_response_speed,
        )

        # Update the agent config with new response speed
        self.agent_config.speed = new_response_speed

        # Send updated session configuration to OpenAI
        await self._send_session_update()

    def _setup_event_handlers(self) -> None:
        """Setup event subscriptions."""
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED,
            self._handle_speech_interruption,
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_STARTED_RESPONSE,
            self._handle_response_started,
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_COMPLETED_RESPONSE,
            self._handle_response_completed,
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_CONFIG_UPDATE_REQUEST,
            self._handle_config_update_request,
        )

    async def _handle_speech_interruption(self) -> None:
        """Handle speech interruption and send truncate message."""
        try:
            item_id = self.current_message_context.item_id
            duration_ms = self.current_message_context.current_duration_ms

            # first message so nothing to truncate
            if not item_id or duration_ms is None:
                return

            self.logger.info("Truncating item %s at %d ms", item_id, duration_ms)

            truncate_event = ConversationItemTruncateEvent(
                item_id=item_id,
                content_index=0,
                audio_end_ms=duration_ms,
            )

            success = await self.ws_manager.send_message(truncate_event)
            if success:
                self.logger.info("Truncate message sent successfully")
            else:
                self.logger.error("Failed to send truncate message")

        except Exception as e:
            self.logger.error(
                "Error handling speech interruption: %s", e, exc_info=True
            )

    async def _handle_response_started(self) -> None:
        """Handle response started."""
        self.queue.set_response_active(True)

    async def _handle_response_completed(self) -> None:
        """Handle response completed."""
        self.queue.set_response_active(False)
        await self.queue.process_queue()
