from __future__ import annotations

from agent.realtime.event_bus import EventBus
from agent.realtime.events.client.conversation_response_create import (
    ConversationResponseCreateEvent,
)
from agent.realtime.messaging.message_manager import RealtimeMessageManager
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


class McpToolHandler(LoggingMixin):
    """
    Handler for MCP (Model Context Protocol) tool events.

    This class listens to MCP events and triggers inference by sending
    ConversationResponseCreate events without actually executing the tools.
    The goal is to let the OpenAI Realtime API handle MCP tool execution
    while we handle the coordination and response generation.
    """

    def __init__(
        self,
        event_bus: EventBus,
        message_manager: RealtimeMessageManager,
        ws_manager: WebSocketManager,
    ):
        self.event_bus = event_bus
        self.message_manager = message_manager
        self.ws_manager = ws_manager

        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_COMPLETED_MCP_TOOL_CALL_RESULT,
            self._handle_mcp_tool_call_completed,
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_FAILED_MCP_TOOL_CALL,
            self._handle_mcp_tool_call_failed,
        )

        self.logger.info("McpToolHandler initialized and subscribed to MCP events")

    async def _handle_mcp_tool_call_completed(self) -> None:
        """
        Handle MCP tool call completed event.

        When an MCP tool call completes successfully, we trigger inference
        to let the assistant process the results and generate a response.
        """
        try:
            self.logger.info(
                "MCP tool call completed - triggering inference for result processing"
            )

            response_event = ConversationResponseCreateEvent.with_instructions(
                "MCP tool call has completed successfully. Please process the results and provide a response to the user."
            )

            # Send the response event to trigger inference
            success = await self.ws_manager.send_message(response_event)

            if success:
                self.logger.debug(
                    "Successfully triggered inference for MCP tool call completion"
                )
            else:
                self.logger.error(
                    "Failed to send ConversationResponseCreate for MCP tool call completion"
                )

        except Exception as e:
            self.logger.error(
                "Error handling MCP tool call completed: %s", e, exc_info=True
            )

    async def _handle_mcp_tool_call_failed(self) -> None:
        """
        Handle MCP tool call failed event.

        When an MCP tool call fails, we trigger inference to let the assistant
        handle the error gracefully and inform the user.
        """
        try:
            self.logger.info(
                "MCP tool call failed - triggering inference for error handling"
            )

            # Create a response event to trigger inference for error handling
            response_event = ConversationResponseCreateEvent.with_instructions(
                "Something went wrong with the MCP tool call. Please inform the user about the issue."
            )

            # Send the response event to trigger inference
            success = await self.ws_manager.send_message(response_event)

            if success:
                self.logger.debug(
                    "Successfully triggered inference for MCP tool call failure"
                )
            else:
                self.logger.error(
                    "Failed to send ConversationResponseCreate for MCP tool call failure"
                )

        except Exception as e:
            self.logger.error(
                "Error handling MCP tool call failed: %s", e, exc_info=True
            )
