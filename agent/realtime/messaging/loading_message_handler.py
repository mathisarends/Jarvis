from agent.realtime.events.client.conversation_item_create import ConversationItemCreateEvent
from agent.realtime.events.client.conversation_response_create import (
    ConversationResponseCreateEvent,
)
from agent.realtime.websocket.websocket_manager import WebSocketManager
from shared.logging_mixin import LoggingMixin


class LoadingMessageHandler(LoggingMixin):
    """Handles loading messages for long-running operations."""

    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager

    async def send_loading_message(self, message: str) -> None:
        """Send loading message to user."""
        try:
            self.logger.info("Sending loading message")

            # Send conversation item
            conversation_item = ConversationItemCreateEvent.assistant_message(message)
            if not await self.ws_manager.send_message(conversation_item):
                self.logger.error("Failed to send loading message conversation item")
                return

            # Trigger response
            response_event = ConversationResponseCreateEvent.with_instructions(
                "Briefly inform the user that the requested tool is running. Keep it short."
            )

            if await self.ws_manager.send_message(response_event):
                self.logger.info("Loading message sent successfully")
            else:
                self.logger.error("Failed to send response.create for loading message")

        except Exception as e:
            self.logger.error("Error sending loading message: %s", e, exc_info=True)
