from agent.config.models import ModelSettings, VoiceSettings
from agent.events import EventBus
from agent.events.schemas import (
    AudioConfig,
    AudioFormat,
    AudioFormatConfig,
    AudioInputConfig,
    AudioOutputConfig,
    ConversationItemTruncatedEvent,
    ConversationResponseCreateEvent,
    RealtimeSessionConfig,
    SessionUpdateEvent,
)
from agent.realtime.current_message_context import CurrentMessageContext
from agent.realtime.messaging.message_queue import MessageQueue
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.state.base import VoiceAssistantEvent
from agent.tools.models import FunctionCallResult
from agent.tools.registry.service import ToolRegistry
from shared.logging_mixin import LoggingMixin


class RealtimeMessageManager(LoggingMixin):
    def __init__(
        self,
        ws_manager: WebSocketManager,
        tool_registry: ToolRegistry,
        model_settings: ModelSettings,
        voice_settings: VoiceSettings,
        event_bus: EventBus,
    ):
        self.ws_manager = ws_manager
        self.event_bus = event_bus
        self.model_settings = model_settings
        self.voice_settings = voice_settings
        self.tool_registry = tool_registry
        self.current_message_context = CurrentMessageContext(self.event_bus)

        self.queue = MessageQueue()
        self.event_bus = event_bus
        self._setup_event_handlers()

    async def initialize_session(self) -> None:
        if not self.ws_manager.is_connected():
            raise RuntimeError("No connection available for session initialization")

        session_config = self._build_config()
        self.logger.info("Sending session update...")
        await self.ws_manager.send_message(session_config)
        self.logger.info("Session initialized successfully")

    async def send_tool_result(self, function_call_result: FunctionCallResult) -> None:
        await self.queue.send_or_queue(
            self._send_tool_result_direct, function_call_result
        )

    async def send_execution_message(self, message: str) -> None:
        await self.queue.send_or_queue(self._send_execution_message_direct, message)

    async def _send_tool_result_direct(
        self, function_call_result: FunctionCallResult
    ) -> None:
        self.logger.info("Sending tool result for '%s'", function_call_result.tool_name)

        await self._send_conversation_item(function_call_result)
        await self._trigger_response(function_call_result)

    async def _send_execution_message_direct(self, message: str) -> None:
        self.logger.info("Sending generator tool update")

        response_event = ConversationResponseCreateEvent.with_instructions(
            f"Say exactly: '{message}'. Do not add any information not in this message."
        )

        await self.ws_manager.send_message(response_event)

    async def _send_conversation_item(self, result: FunctionCallResult) -> None:
        conversation_item = result.to_conversation_item()
        await self.ws_manager.send_message(conversation_item)

    async def _trigger_response(self, result: FunctionCallResult) -> None:
        response_event = ConversationResponseCreateEvent.with_instructions(
            result.response_instruction
            or "Process the tool result and provide a helpful response."
        )
        await self.ws_manager.send_message(response_event)

    async def _send_session_update(self):
        session_config = self._build_config()
        self.logger.info("Sending session update with new config...")

        await self.ws_manager.send_message(session_config)

    def _build_config(self) -> SessionUpdateEvent:
        audio_config = AudioConfig(
            output=AudioOutputConfig(
                format=AudioFormatConfig(type=AudioFormat.PCM16),
                speed=self.voice_settings.speech_speed,
                voice=self.voice_settings.assistant_voice,
            ),
            input=AudioInputConfig(),
        )

        return SessionUpdateEvent(
            session=RealtimeSessionConfig(
                model=self.model_settings.model,
                instructions=self.model_settings.instructions,
                audio=audio_config,
                output_modalities=["audio"],
                tools=self.tool_registry.get_openai_schema(),
            ),
        )

    async def _handle_config_update_request(self, new_response_speed: float):
        self.logger.info(
            "Received config update request - New response speed: %.2f",
            new_response_speed,
        )

        self.voice_settings.speech_speed = new_response_speed

        await self._send_session_update()

    def _setup_event_handlers(self) -> None:
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
        item_id = self.current_message_context.item_id
        duration_ms = self.current_message_context.current_duration_ms

        # first message so nothing to truncate
        if not item_id or duration_ms is None:
            return

        self.logger.info("Truncating item %s at %d ms", item_id, duration_ms)

        truncate_event = ConversationItemTruncatedEvent(
            item_id=item_id,
            content_index=0,
            audio_end_ms=duration_ms,
        )

        await self.ws_manager.send_message(truncate_event)

    async def _handle_response_started(self) -> None:
        self.queue.set_response_active(True)

    async def _handle_response_completed(self) -> None:
        self.queue.set_response_active(False)
        await self.queue.process_queue()
