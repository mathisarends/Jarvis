from collections.abc import Callable
from typing import Any

from rtvoice.events import EventBus
from rtvoice.events.schemas import (
    ConversationItemTruncatedEvent,
    ErrorEvent,
    ResponseOutputAudioDeltaEvent,
    SessionCreatedEvent,
)
from rtvoice.events.schemas.base import RealtimeServerEvent
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.state.base import VoiceAssistantEvent
from rtvoice.tools.models import FunctionCallItem
from rtvoice.transcription.models import (
    InputAudioTranscriptionCompleted,
    ResponseOutputAudioTranscriptDone,
)


class EventDispatcher(LoggingMixin):
    def __init__(self, event_bus: EventBus):
        self._event_bus = event_bus
        self._event_handlers = self._build_event_handler_map()
        self._ignored_events = self._build_ignored_events_set()

    def dispatch_event(self, data: dict[str, Any]) -> None:
        event_type_str = data.get("type", "")

        if not event_type_str:
            self.logger.warning("Received event without type field: %s", data)
            return

        event_type = self._parse_event_type(event_type_str, data)
        if not event_type:
            return

        if event_type in self._ignored_events:
            self.logger.debug("Received ignored event: %s", event_type_str)
            return

        self._route_to_handler(event_type, event_type_str, data)

    def _build_event_handler_map(
        self,
    ) -> dict[RealtimeServerEvent, Callable[[dict[str, Any]], None]]:
        return {
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED: self._handle_user_speech_started,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED: self._handle_user_speech_stopped,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA: self._handle_audio_chunk_received,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED: self._handle_user_transcript_completed,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE: self._handle_assistant_transcript_completed,
            RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE: self._handle_response_function_call_completed,
            RealtimeServerEvent.SESSION_CREATED: self._handle_session_created,
            RealtimeServerEvent.SESSION_UPDATED: self._handle_session_updated,
            RealtimeServerEvent.CONVERSATION_ITEM_TRUNCATED: self._handle_speech_interruption,
            RealtimeServerEvent.MCP_CALL_ARGUMENTS_DONE: self._handle_mcp_call_arguments_done,
            RealtimeServerEvent.RESPONSE_MCP_CALL_COMPLETED: self._handle_mcp_call_completed,
            RealtimeServerEvent.RESPONSE_MCP_CALL_FAILED: self._handle_mcp_call_failed,
            RealtimeServerEvent.RESPONSE_CREATED: self._handle_response_created,
            RealtimeServerEvent.RESPONSE_DONE: self._handle_response_done,
            RealtimeServerEvent.ERROR: self._handle_api_error,
        }

    def _build_ignored_events_set(self) -> set[RealtimeServerEvent]:
        return {
            RealtimeServerEvent.TRANSCRIPTION_SESSION_UPDATED,
            RealtimeServerEvent.CONVERSATION_CREATED,
            RealtimeServerEvent.CONVERSATION_DELETED,
            RealtimeServerEvent.CONVERSATION_ITEM_CREATED,
            RealtimeServerEvent.CONVERSATION_ITEM_ADDED,
            RealtimeServerEvent.CONVERSATION_ITEM_DONE,
            RealtimeServerEvent.CONVERSATION_ITEM_RETRIEVED,
            RealtimeServerEvent.CONVERSATION_ITEM_DELETED,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_SEGMENT,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_COMMITTED,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_CLEARED,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_TIMEOUT_TRIGGERED,
            RealtimeServerEvent.RESPONSE_OUTPUT_ITEM_ADDED,
            RealtimeServerEvent.RESPONSE_OUTPUT_ITEM_DONE,
            RealtimeServerEvent.RESPONSE_CONTENT_PART_ADDED,
            RealtimeServerEvent.RESPONSE_CONTENT_PART_DONE,
            RealtimeServerEvent.RESPONSE_OUTPUT_TEXT_DELTA,
            RealtimeServerEvent.RESPONSE_OUTPUT_TEXT_DONE,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DONE,
            RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA,
            RealtimeServerEvent.MCP_CALL_ARGUMENTS_DELTA,
            RealtimeServerEvent.MCP_LIST_TOOLS_IN_PROGRESS,
            RealtimeServerEvent.MCP_LIST_TOOLS_COMPLETED,
            RealtimeServerEvent.MCP_LIST_TOOLS_FAILED,
            RealtimeServerEvent.RESPONSE_MCP_CALL_IN_PROGRESS,
            RealtimeServerEvent.RESPONSE_MCP_CALL_FAILED,
            RealtimeServerEvent.RATE_LIMITS_UPDATED,
        }

    def _parse_event_type(
        self, event_type_str: str, data: dict[str, Any]
    ) -> RealtimeServerEvent | None:
        try:
            return RealtimeServerEvent(event_type_str)
        except ValueError:
            if event_type_str == "error":
                self._handle_api_error(data)
            else:
                self.logger.warning("Unknown OpenAI event type: %s", event_type_str)
            return None

    def _route_to_handler(
        self, event_type: RealtimeServerEvent, event_type_str: str, data: dict[str, Any]
    ) -> None:
        handler = self._event_handlers.get(event_type)
        if not handler:
            self.logger.debug("No handler registered for event: %s", event_type_str)
            return

        handler(data)

    def _handle_user_speech_started(self, data: dict[str, Any]) -> None:
        self.logger.debug("User started speaking")
        self._event_bus.publish_sync(VoiceAssistantEvent.USER_STARTED_SPEAKING)

    def _handle_user_speech_stopped(self, data: dict[str, Any]) -> None:
        self.logger.debug("User speech ended")
        self._event_bus.publish_sync(VoiceAssistantEvent.USER_SPEECH_ENDED)

    def _handle_audio_chunk_received(self, data: dict[str, Any]) -> None:
        audio_data = ResponseOutputAudioDeltaEvent.model_validate(data)
        if not audio_data.delta:
            self.logger.warning("Received empty audio delta")
            return

        self._event_bus.publish_sync(
            VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED, audio_data
        )

    def _handle_user_transcript_completed(self, data: dict[str, Any]) -> None:
        payload = InputAudioTranscriptionCompleted.model_validate(data)
        self._event_bus.publish_sync(
            VoiceAssistantEvent.USER_TRANSCRIPT_COMPLETED, payload
        )

    def _handle_assistant_transcript_completed(self, data: dict[str, Any]) -> None:
        transcript = ResponseOutputAudioTranscriptDone.model_validate(data)
        self._event_bus.publish_sync(
            VoiceAssistantEvent.ASSISTANT_TRANSCRIPT_COMPLETED, transcript
        )

    def _handle_response_function_call_completed(self, data: dict[str, Any]) -> None:
        function_call_item = FunctionCallItem.model_validate(data)
        self.logger.info(
            "Function call initiated - Tool: %s, Args: %s",
            function_call_item.name,
            function_call_item.arguments,
        )
        self._event_bus.publish_sync(
            VoiceAssistantEvent.ASSISTANT_STARTED_TOOL_CALL, function_call_item
        )

    def _handle_session_updated(self, data: dict[str, Any]) -> None:
        pass

    def _handle_speech_interruption(self, data: dict[str, Any]) -> None:
        truncate_event = ConversationItemTruncatedEvent.model_validate(data)
        self.logger.info(
            "Conversation item truncated - Item ID: %s, Audio End MS: %s",
            truncate_event.item_id,
            truncate_event.audio_end_ms,
        )
        self._event_bus.publish_sync(
            VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED, truncate_event
        )

    def _handle_mcp_call_arguments_done(self, data: dict[str, Any]) -> None:
        self.logger.info("MCP call arguments completed, starting MCP tool execution")
        self.logger.debug("MCP call arguments data: %s", data)
        self._event_bus.publish_sync(
            VoiceAssistantEvent.ASSISTANT_STARTED_MCP_TOOL_CALL
        )

    def _handle_mcp_call_completed(self, data: dict[str, Any]) -> None:
        self.logger.info("MCP tool call completed successfully")
        self._event_bus.publish_sync(
            VoiceAssistantEvent.ASSISTANT_COMPLETED_MCP_TOOL_CALL_RESULT
        )

    def _handle_mcp_call_failed(self, data: dict[str, Any]) -> None:
        error_details = data.get("error", {})
        error_message = error_details.get("message", "Unknown MCP error")
        error_type = error_details.get("type", "unknown")

        self.logger.error(
            "MCP tool call failed: %s (type: %s)", error_message, error_type
        )
        self.logger.debug("MCP failure data: %s", data)
        self._event_bus.publish_sync(
            VoiceAssistantEvent.ASSISTANT_FAILED_MCP_TOOL_CALL, data
        )

    def _handle_response_created(self, data: dict[str, Any]) -> None:
        self.logger.debug("Assistant response started")
        self._event_bus.publish_sync(VoiceAssistantEvent.ASSISTANT_STARTED_RESPONSE)

    def _handle_response_done(self, data: dict[str, Any]) -> None:
        self.logger.debug("Assistant response completed")
        self._event_bus.publish_sync(VoiceAssistantEvent.ASSISTANT_COMPLETED_RESPONSE)

    def _handle_api_error(self, data: dict[str, Any]) -> None:
        error_event = ErrorEvent.model_validate(data)
        self.logger.error(
            "OpenAI API error: %s (type: %s, code: %s)",
            error_event.error.message,
            error_event.error.type,
            error_event.error.code,
        )
        self._event_bus.publish_sync(VoiceAssistantEvent.ERROR_OCCURRED, error_event)

    def _handle_session_created(self, data: dict[str, Any]) -> None:
        session_event = SessionCreatedEvent.model_validate(data)
        session_config = session_event.session

        self.logger.debug("Full session config: %s", session_config.model_dump())
