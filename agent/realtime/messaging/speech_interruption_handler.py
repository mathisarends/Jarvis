from agent.realtime.current_message_context import CurrentMessageContext
from agent.realtime.event_types import RealtimeClientEvent
from agent.realtime.views import ConversationItemTruncateEvent
from agent.realtime.websocket.websocket_manager import WebSocketManager
from shared.logging_mixin import LoggingMixin


class SpeechInterruptionHandler(LoggingMixin):
    """Handles speech interruption events."""

    def __init__(
        self,
        ws_manager: WebSocketManager,
        current_message_context: CurrentMessageContext,
    ):
        self.ws_manager = ws_manager
        self.current_message_context = current_message_context

    async def handle_interruption(self) -> None:
        """Handle speech interruption and send truncate message."""
        try:
            item_id = self.current_message_context.item_id
            duration_ms = self.current_message_context.current_duration_ms

            # first messageg so nothing to truncate
            if not item_id or duration_ms is None:
                return

            self.logger.info("Truncating item %s at %d ms", item_id, duration_ms)

            truncate_event = ConversationItemTruncateEvent(
                type=RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE,
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
