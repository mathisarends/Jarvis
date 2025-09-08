from agent.realtime.tools.views import FunctionCallResult
from agent.realtime.events.client.conversation_response_create import (
    ConversationResponseCreateEvent,
)
from agent.realtime.websocket.websocket_manager import WebSocketManager
from shared.logging_mixin import LoggingMixin


class ToolMessageHandler(LoggingMixin):
    """Handles tool execution results and generator updates."""

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

    async def send_execution_message(self, message: str) -> None:
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
