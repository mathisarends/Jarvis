from agent.realtime.events.client.conversation_item_truncate import (
    ConversationItemTruncateEvent,
)
from agent.realtime.events.client.conversation_response_create import (
    ConversationResponseCreateEvent,
)
from agent.realtime.events.client.input_audio_buffer_append import (
    InputAudioBufferAppendEvent,
)
from agent.realtime.events.client.session_update import SessionUpdateEvent
from agent.realtime.tools.views import FunctionCallResult
from agent.realtime.websocket.websocket_manager import WebSocketManager
from shared.logging_mixin import LoggingMixin
import base64


class RealtimeMessageSender(LoggingMixin):
    """
    Einfacher Message Sender für Realtime-Nachrichten.
    Verantwortlich nur für das direkte Versenden von Nachrichten.
    """

    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager

    async def send_session_update(self, session_config: SessionUpdateEvent) -> bool:
        """Sendet Session-Update an OpenAI API."""
        try:
            self.logger.info("Sending session update...")
            success = await self.ws_manager.send_message(session_config)

            if success:
                self.logger.info("Session update sent successfully")
            else:
                self.logger.error("Failed to send session update")

            return success
        except Exception as e:
            self.logger.error("Error sending session update: %s", e)
            return False

    async def send_tool_result(self, function_call_result: FunctionCallResult) -> bool:
        """Sendet Tool-Ergebnis an OpenAI API."""
        try:
            self.logger.info(
                "Sending tool result for '%s'", function_call_result.tool_name
            )

            conversation_item = function_call_result.to_conversation_item()
            success = await self.ws_manager.send_message(conversation_item)

            if not success:
                self.logger.error(
                    "Failed to send function_call_output for '%s'",
                    function_call_result.tool_name,
                )

            return success
        except Exception as e:
            self.logger.error(
                "Error sending tool result for '%s': %s",
                function_call_result.tool_name,
                e,
            )
            return False

    async def send_response_create(self, instructions: str = None) -> bool:
        """Sendet Response-Create Event."""
        try:
            self.logger.info("Sending response create event")

            response_event = ConversationResponseCreateEvent.with_instructions(
                instructions
                or "Process the previous information and provide a helpful response."
            )

            success = await self.ws_manager.send_message(response_event)

            if success:
                self.logger.info("Response create event sent successfully")
            else:
                self.logger.error("Failed to send response create event")

            return success
        except Exception as e:
            self.logger.error("Error sending response create: %s", e)
            return False

    async def send_message(self, truncate_event: ConversationItemTruncateEvent) -> bool:
        """Sendet Truncate-Event für Unterbrechungen."""
        try:
            self.logger.info(
                "Sending truncate for item %s at %d ms",
                truncate_event.item_id,
                truncate_event.audio_end_ms,
            )

            success = await self.ws_manager.send_message(truncate_event)

            if success:
                self.logger.info("Truncate message sent successfully")
            else:
                self.logger.error("Failed to send truncate message")

            return success
        except Exception as e:
            self.logger.error("Error sending truncate: %s", e)
            return False

    async def send_execution_update(self, message: str) -> bool:
        """Sendet Ausführungs-Update für Generator-Tools."""
        try:
            self.logger.info("Sending execution update")

            response_event = ConversationResponseCreateEvent.with_instructions(
                f"Say exactly: '{message}'. Do not add any information not in this message."
            )

            success = await self.ws_manager.send_message(response_event)

            if success:
                self.logger.info("Execution update sent successfully")
            else:
                self.logger.error("Failed to send execution update")

            return success
        except Exception as e:
            self.logger.error("Error sending execution update: %s", e)
            return False

    async def send_audio_chunk(self, audio_chunk: bytes) -> bool:
        """Sendet ein Audio-Chunk als base64-kodiertes Event."""
        try:
            base64_audio_data = base64.b64encode(audio_chunk).decode("utf-8")
            input_audio_buffer_append_event = InputAudioBufferAppendEvent.from_audio(
                base64_audio_data
            )
            success = await self.ws_manager.send_message(
                input_audio_buffer_append_event
            )

            if not success:
                self.logger.error("Failed to send audio chunk")

            return success
        except Exception as e:
            self.logger.error("Error sending audio chunk: %s", e)
            return False

    def is_connected(self) -> bool:
        """Prüft ob WebSocket-Verbindung aktiv ist."""
        return self.ws_manager.is_connected()
