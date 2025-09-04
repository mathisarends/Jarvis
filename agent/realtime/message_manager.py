from agent.realtime.current_message_context import CurrentMessageContext
from agent.realtime.event_bus import EventBus
from agent.realtime.event_types import RealtimeClientEvent
from agent.realtime.tools.views import FunctionCallResult
from agent.realtime.views import ConversationItemTruncateEvent
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


class RealtimeMessageManager(LoggingMixin):

    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager

        self.event_bus = EventBus()
        self.current_message_timer = CurrentMessageContext()

        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT,
            self._handle_tool_result,
        )

        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED,
            self._handle_speech_interruption,
        )

    async def _handle_tool_result(
        self, event: VoiceAssistantEvent, data: FunctionCallResult
    ) -> None:
        """Handle tool execution results and send them back to OpenAI Realtime API"""
        try:
            self.logger.info(
                "Received tool result for '%s', sending to Realtime API", data.tool_name
            )

            # 1) Tool-Output als conversation item posten
            conversation_item = data.to_conversation_item()
            ok_item = await self.ws_manager.send_message(conversation_item)
            if not ok_item:
                self.logger.error(
                    "Failed to send function_call_output for '%s'", data.tool_name
                )
                return

            self.logger.info(
                "function_call_output for '%s' sent. Triggering response.create...",
                data.tool_name,
            )

            # 2) Modell anstoßen, das Ergebnis zu verwenden
            #    (Du kannst instructions optional leer lassen oder kurz kontextualisieren)
            response_create = {
                "type": "response.create",
            }

            ok_resp = await self.ws_manager.send_message(response_create)
            if not ok_resp:
                self.logger.error("Failed to send response.create")
                return

            self.logger.info("response.create sent successfully")

        except Exception as e:
            self.logger.error(
                "Error handling tool result for '%s': %s",
                data.tool_name,
                e,
                exc_info=True,
            )

    async def _handle_speech_interruption(
        self, event: VoiceAssistantEvent, data=None
    ) -> None:
        """Handle speech interruption events and send truncate message."""
        try:
            # Get current item_id and duration from current_message_context
            item_id = self.current_message_timer.item_id
            duration_ms = self.current_message_timer.current_duration_ms

            if not item_id:
                self.logger.warning(
                    "Speech interrupted but no current item_id available"
                )
                return

            if duration_ms is None:
                self.logger.warning(
                    "Speech interrupted but no current duration available"
                )
                return

            self.logger.info(
                "Speech interrupted - truncating item %s at %d ms", item_id, duration_ms
            )

            truncate_event = ConversationItemTruncateEvent(
                type=RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE,
                item_id=item_id,
                content_index=0,
                audio_end_ms=duration_ms,
            )

            success = await self.ws_manager.send_message(
                truncate_event.model_dump(exclude_unset=True)
            )

            if success:
                self.logger.info(
                    "Truncate message sent successfully for item %s", item_id
                )
            else:
                self.logger.error(
                    "Failed to send truncate message for item %s", item_id
                )

        except Exception as e:
            self.logger.error(
                "Error handling speech interruption: %s", e, exc_info=True
            )
