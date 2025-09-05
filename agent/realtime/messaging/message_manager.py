from agent.config.views import VoiceAssistantConfig
from agent.realtime.current_message_context import CurrentMessageContext
from agent.realtime.event_bus import EventBus
from agent.realtime.messaging.loading_message_handler import LoadingMessageHandler
from agent.realtime.messaging.message_queue import MessageQueue
from agent.realtime.messaging.session_manager import SessionManager
from agent.realtime.messaging.speech_interruption_handler import (
    SpeechInterruptionHandler,
)
from agent.realtime.messaging.tool_result_handler import ToolResultHandler
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.views import FunctionCallResult
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


class RealtimeMessageManager(LoggingMixin):
    """
    Clean orchestrator for realtime message management.
    Delegates specific responsibilities to specialized components.
    """

    def __init__(
        self,
        ws_manager: WebSocketManager,
        tool_registry: ToolRegistry,
        voice_assistant_config: VoiceAssistantConfig,
    ):
        self.ws_manager = ws_manager

        # Initialize specialized components
        self.queue = MessageQueue()
        self.session_manager = SessionManager(
            voice_assistant_config.agent, tool_registry, ws_manager
        )
        self.tool_handler = ToolResultHandler(ws_manager)
        self.loading_handler = LoadingMessageHandler(ws_manager)
        self.interruption_handler = SpeechInterruptionHandler(
            ws_manager, CurrentMessageContext()
        )

        # Setup event handling
        self.event_bus = EventBus()
        self._setup_event_handlers()

    async def initialize_session(self) -> bool:
        """Initialize session with OpenAI API."""
        return await self.session_manager.initialize()

    async def send_tool_result(self, function_call_result: FunctionCallResult) -> None:
        """Send tool result (queued if response active)."""
        await self.queue.send_or_queue(
            self.tool_handler.send_result, function_call_result
        )

    async def send_loading_message(self, message: str) -> None:
        """Send loading message (queued if response active)."""
        await self.queue.send_or_queue(
            self.loading_handler.send_loading_message, message
        )

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

    async def _handle_speech_interruption(
        self, event: VoiceAssistantEvent, data=None
    ) -> None:
        """Handle speech interruption."""
        await self.interruption_handler.handle_interruption()

    async def _handle_response_started(
        self, event: VoiceAssistantEvent, data=None
    ) -> None:
        """Handle response started."""
        self.queue.set_response_active(True)

    async def _handle_response_completed(
        self, event: VoiceAssistantEvent, data=None
    ) -> None:
        """Handle response completed."""
        self.queue.set_response_active(False)
        await self.queue.process_queue()
