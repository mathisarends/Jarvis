from __future__ import annotations

from typing import TYPE_CHECKING

from rtvoice.events import EventBus
from rtvoice.events.schemas import ConversationResponseCreateEvent
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent

if TYPE_CHECKING:
    from rtvoice.realtime.messaging.message_manager import RealtimeMessageManager
    from rtvoice.realtime.websocket.websocket_manager import WebSocketManager


class RemoteMcpToolEventListener(LoggingMixin):
    def __init__(
        self,
        event_bus: EventBus,
        message_manager: RealtimeMessageManager,
        ws_manager: WebSocketManager,
    ):
        self._event_bus = event_bus
        self._message_manager = message_manager
        self._ws_manager = ws_manager

        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_COMPLETED_MCP_TOOL_CALL_RESULT,
            self._handle_remote_mcp_tool_call_completed,
        )
        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_FAILED_MCP_TOOL_CALL,
            self._handle_remote_mcp_tool_call_failed,
        )

    async def _handle_remote_mcp_tool_call_completed(self) -> None:
        response_event = ConversationResponseCreateEvent.with_instructions(
            "MCP tool call has completed successfully. Please process the results "
            "and provide a response to the user."
        )

        await self._ws_manager.send_message(response_event)

    async def _handle_remote_mcp_tool_call_failed(self) -> None:
        response_event = ConversationResponseCreateEvent.with_instructions(
            "Something went wrong with the MCP tool call. Please inform the user "
            "about the issue."
        )

        await self._ws_manager.send_message(response_event)
