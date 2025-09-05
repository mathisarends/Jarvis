from agent.config.views import VoiceAssistantConfig
from agent.realtime.current_message_context import CurrentMessageContext
from agent.realtime.event_bus import EventBus
from agent.realtime.event_types import RealtimeClientEvent
from agent.realtime.events.conversation_response_create import (
    ConversationResponseCreateEvent,
    ResponseInstructions,
)
from agent.realtime.tools.views import FunctionCallResult
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.views import (
    ConversationItemTruncateEvent,
    SessionUpdateEvent,
    AudioConfig,
    AudioOutputConfig,
    AudioFormatConfig,
    AudioFormat,
    SessionConfig,
)
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin
from typing import Any


# Hier brauche ich utils wie send_converation_item oder so
class RealtimeMessageManager(LoggingMixin):

    def __init__(
        self,
        ws_manager: WebSocketManager,
        tool_registry: ToolRegistry,
        voice_assistant_config: VoiceAssistantConfig,
    ):
        self.ws_manager = ws_manager
        self.tool_registry = tool_registry

        self.realtime_model = voice_assistant_config.agent.model
        self.instructions = voice_assistant_config.agent.instructions
        self.voice = voice_assistant_config.agent.voice
        self.temperature = voice_assistant_config.agent.temperature
        self.max_response_output_tokens = (
            voice_assistant_config.agent.max_response_output_tokens
        )
        self.input_audio_transcription = (
            voice_assistant_config.agent.transcription
        )  # this field is not working for now (altough described in documentary)
        self.input_audio_noise_reduction = (
            voice_assistant_config.agent.input_audio_noise_reduction
        )  # not working as well

        self.event_bus = EventBus()
        self.current_message_timer = CurrentMessageContext()

        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED,
            self._handle_speech_interruption,
        )

    async def send_tool_result(self, function_call_result: FunctionCallResult) -> None:
        """Handle tool execution results and send them back to OpenAI Realtime API"""
        try:
            self.logger.info(
                "Received tool result for '%s', sending to Realtime API",
                function_call_result.tool_name,
            )

            # 1) Tool-Output als conversation item posten
            conversation_item = function_call_result.to_conversation_item()
            ok_item = await self.ws_manager.send_message(conversation_item)
            if not ok_item:
                self.logger.error(
                    "Failed to send function_call_output for '%s'",
                    function_call_result.tool_name,
                )
                return

            self.logger.info(
                "function_call_output for '%s' sent. Triggering response.create...",
                function_call_result.tool_name,
            )

            conversation_response_create_event = ConversationResponseCreateEvent(
                type=RealtimeClientEvent.RESPONSE_CREATE,
                response=ResponseInstructions(
                    instructions=function_call_result.result_context
                ),
            )

            response_dict = conversation_response_create_event.model_dump(
                exclude_unset=True
            )

            ok_resp = await self.ws_manager.send_message(response_dict)
            if not ok_resp:
                self.logger.error("Failed to send response.create")
                return

            self.logger.info("response.create sent successfully")

        except Exception as e:
            self.logger.error(
                "Error handling tool result for '%s': %s",
                function_call_result.tool_name,
                e,
                exc_info=True,
            )

    async def initialize_session(self) -> bool:
        """
        Initializes a session with the OpenAI API.
        """
        if not self.ws_manager.is_connected():
            self.logger.error("No connection available for session initialization")
            return False

        session_update = self._build_session_config()

        try:
            self.logger.info("Sending session update...")
            success = await self.ws_manager.send_message(session_update)

            if success:
                self.logger.info("Session update sent successfully")
                return True

            self.logger.error("Failed to send session update")
            return False

        except Exception as e:
            self.logger.error("Error initializing session: %s", e)
            return False

    async def send_loading_message_for_long_running_tool_call(
        self, loading_message: str
    ) -> None:
        try:
            self.logger.info("Sending loading message for long-running tool call")
            conversation_item = {
                "type": RealtimeClientEvent.CONVERSATION_ITEM_CREATE,
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": loading_message}],
                },
            }

            ok_item = await self.ws_manager.send_message(conversation_item)
            if not ok_item:
                self.logger.error("Failed to send loading message conversation item")
                return

            self.logger.info("Loading message sent. Triggering response.create...")

            conversation_response_create_event = ConversationResponseCreateEvent(
                type=RealtimeClientEvent.RESPONSE_CREATE,
            )

            response_dict = conversation_response_create_event.model_dump(
                exclude_unset=True
            )
            ok_resp = await self.ws_manager.send_message(response_dict)
            if not ok_resp:
                self.logger.error("Failed to send response.create for loading message")
                return

            self.logger.info("Response.create sent successfully for loading message")

        except Exception as e:
            self.logger.error(
                "Error sending loading message for long-running tool call: %s",
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

    def _build_session_config(self) -> dict[str, Any]:
        """
        Creates the session configuration for the OpenAI API.
        Uses fully typed Pydantic models based on the official API documentation.
        """
        # Create audio configuration with nested format objects
        audio_config = AudioConfig(
            output=AudioOutputConfig(
                format=AudioFormatConfig(type=AudioFormat.PCM16),
                voice=self.voice,
            )
        )

        session_config = SessionUpdateEvent(
            type=RealtimeClientEvent.SESSION_UPDATE,
            session=SessionConfig(
                type="realtime",
                model=self.realtime_model,
                instructions=self.instructions,
                audio=audio_config,
                output_modalities=["audio"],
                max_output_tokens=self.max_response_output_tokens,
                tools=self.tool_registry.get_openai_schema(),
            ),
        )

        return session_config.model_dump(exclude_unset=True)
