from agent.realtime.tools.views import FunctionCallResult
from agent.realtime.events.client.conversation_response_create import (
    ConversationResponseCreateEvent,
)
from agent.realtime.websocket.websocket_manager import WebSocketManager
from shared.logging_mixin import LoggingMixin


class ToolResultHandler(LoggingMixin):
    """Handles tool execution results."""

    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager

    async def send_result(self, function_call_result: FunctionCallResult) -> None:
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
            result.result_context
        )

        success = await self.ws_manager.send_message(response_event)
        if success:
            self.logger.info("Response.create sent successfully")
        else:
            self.logger.error("Failed to send response.create")
