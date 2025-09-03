from __future__ import annotations
from typing import Any, Callable
from pydantic import ValidationError

from agent.realtime.event_types import RealtimeServerEvent
from agent.realtime.event_bus import EventBus
from agent.state.base import VoiceAssistantEvent
from agent.realtime.transcription.views import (
    InputAudioTranscriptionCompleted,
    ResponseOutputAudioTranscriptDone,
)
from agent.realtime.views import ResponseOutputAudioDelta, ErrorEvent
from shared.logging_mixin import LoggingMixin
from shared.singleton_decorator import singleton


@singleton
class RealtimeEventDispatcher(LoggingMixin):
    """
    Dispatcher für OpenAI Realtime API Events.
    Mapt eingehende API Events auf interne VoiceAssistantEvents und published diese über den EventBus.
    Bietet saubere Trennung zwischen WebSocket-Handling und Event-Verarbeitung.
    """

    def __init__(self):
        self.event_bus = EventBus()

        # Event mapping: OpenAI API event type -> handler method
        self.event_handlers: dict[RealtimeServerEvent, Callable[[dict[str, Any]], None]] = {
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED: self.handle_user_speech_started,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED: self.handle_user_speech_stopped,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA: self.handle_audio_chunk_received,
            RealtimeServerEvent.RESPONSE_DONE: self.handle_response_completed,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED: self.handle_user_transcript_completed,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE: self.handle_assistant_transcript_completed,
            RealtimeServerEvent.ERROR: self.handle_api_error,
        }

        # Events die wir loggen aber nicht weiter verarbeiten
        self.ignored_events = {
            # Session events
            RealtimeServerEvent.SESSION_CREATED,
            RealtimeServerEvent.SESSION_UPDATED,
            RealtimeServerEvent.TRANSCRIPTION_SESSION_CREATED,
            RealtimeServerEvent.TRANSCRIPTION_SESSION_UPDATED,
            # Conversation events
            RealtimeServerEvent.CONVERSATION_CREATED,
            RealtimeServerEvent.CONVERSATION_DELETED,
            RealtimeServerEvent.CONVERSATION_ITEM_CREATED,
            RealtimeServerEvent.CONVERSATION_ITEM_ADDED,
            RealtimeServerEvent.CONVERSATION_ITEM_DONE,
            RealtimeServerEvent.CONVERSATION_ITEM_RETRIEVED,
            RealtimeServerEvent.CONVERSATION_ITEM_TRUNCATED,
            RealtimeServerEvent.CONVERSATION_ITEM_DELETED,
            # Input audio transcription events
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_SEGMENT,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED,
            # Input audio buffer events
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_COMMITTED,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_CLEARED,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_TIMEOUT_TRIGGERED,
            # Response events
            RealtimeServerEvent.RESPONSE_CREATED,
            RealtimeServerEvent.RESPONSE_DONE,
            # Response output events
            RealtimeServerEvent.RESPONSE_OUTPUT_ITEM_ADDED,
            RealtimeServerEvent.RESPONSE_OUTPUT_ITEM_DONE,
            RealtimeServerEvent.RESPONSE_CONTENT_PART_ADDED,
            RealtimeServerEvent.RESPONSE_CONTENT_PART_DONE,
            # Text output events
            RealtimeServerEvent.RESPONSE_OUTPUT_TEXT_DELTA,
            RealtimeServerEvent.RESPONSE_OUTPUT_TEXT_DONE,
            # Audio transcript events
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE,
            # Audio output events
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DONE,
            # Function calling events
            RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA,
            RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE,
            # MCP events
            RealtimeServerEvent.MCP_CALL_ARGUMENTS_DELTA,
            RealtimeServerEvent.MCP_CALL_ARGUMENTS_DONE,
            RealtimeServerEvent.MCP_LIST_TOOLS_IN_PROGRESS,
            RealtimeServerEvent.MCP_LIST_TOOLS_COMPLETED,
            RealtimeServerEvent.MCP_LIST_TOOLS_FAILED,
            RealtimeServerEvent.RESPONSE_MCP_CALL_IN_PROGRESS,
            RealtimeServerEvent.RESPONSE_MCP_CALL_COMPLETED,
            RealtimeServerEvent.RESPONSE_MCP_CALL_FAILED,
            # Rate limits
            RealtimeServerEvent.RATE_LIMITS_UPDATED,
            # Error events
            RealtimeServerEvent.ERROR,
        }

    def dispatch_event(self, data: dict[str, Any]) -> None:
        """
        Hauptmethode für das Dispatching von OpenAI Realtime API Events.
        Routet Events basierend auf dem event type zu den entsprechenden Handlern.
        """
        event_type_str = data.get("type", "")

        if not event_type_str:
            self.logger.warning("Received event without type field: %s", data)
            return

        # Try to convert to enum for handler lookup
        try:
            event_type_enum = RealtimeServerEvent(event_type_str)
        except ValueError:
            event_type_enum = None

        # Prüfe ob wir einen Handler für dieses Event haben
        handler = None
        if event_type_enum:
            handler = self.event_handlers.get(event_type_enum)
        if not handler:
            # Fallback to string lookup for backward compatibility
            handler = self.event_handlers.get(event_type_str)
        if handler:
            try:
                handler(data)
            except Exception as e:
                self.logger.error("Error handling event %s: %s", event_type_str, e)

        elif event_type_enum and event_type_enum in self.ignored_events:
            self.logger.debug("Received ignored event: %s", event_type_str)

        else:
            self.logger.warning("Unknown OpenAI event type: %s", event_type_str)

    # Event Handler Methods - Mapping von OpenAI Events zu VoiceAssistantEvents

    def handle_user_speech_started(self, data: dict[str, Any]) -> None:
        """User started speaking -> USER_STARTED_SPEAKING"""
        self.logger.debug("User started speaking")
        self.event_bus.publish_sync(VoiceAssistantEvent.USER_STARTED_SPEAKING)

    def handle_user_speech_stopped(self, data: dict[str, Any]) -> None:
        """User stopped speaking -> USER_SPEECH_ENDED"""
        self.logger.debug("User speech ended")
        self.event_bus.publish_sync(VoiceAssistantEvent.USER_SPEECH_ENDED)

    def handle_audio_chunk_received(self, data: dict[str, Any]) -> None:
        """Audio chunk received -> AUDIO_CHUNK_RECEIVED"""
        try:
            audio_data = ResponseOutputAudioDelta.model_validate(data)
            if not audio_data.delta:
                self.logger.warning("Received empty audio delta")
                return

            self.event_bus.publish_sync(
                VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED, audio_data.delta
            )
        except ValidationError as e:
            self.logger.warning("Invalid audio delta payload: %s", e)

    def handle_response_completed(self, data: dict[str, Any]) -> None:
        """Assistant response completed -> ASSISTANT_RESPONSE_COMPLETED"""
        self.logger.debug("Assistant response completed")
        self.event_bus.publish_sync(VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED)

    def handle_user_transcript_completed(self, data: dict[str, Any]) -> None:
        """User transcript completed -> USER_TRANSCRIPT_COMPLETED"""
        try:
            payload = InputAudioTranscriptionCompleted.model_validate(data)
            self.event_bus.publish_sync(
                VoiceAssistantEvent.USER_TRANSCRIPT_COMPLETED, payload
            )
        except ValidationError as e:
            self.logger.warning("Invalid user transcript payload: %s", e)

    def handle_assistant_transcript_completed(self, data: dict[str, Any]) -> None:
        """Assistant transcript completed -> ASSISTANT_TRANSCRIPT_COMPLETED"""
        try:
            payload = ResponseOutputAudioTranscriptDone.model_validate(data)
            self.event_bus.publish_sync(
                VoiceAssistantEvent.ASSISTANT_TRANSCRIPT_COMPLETED, payload
            )
        except ValidationError as e:
            self.logger.warning("Invalid assistant transcript payload: %s", e)

    def handle_api_error(self, data: dict[str, Any]) -> None:
        """API error -> ERROR_OCCURRED"""
        try:
            error_event = ErrorEvent.model_validate(data)
            self.logger.error(
                "OpenAI API error [%s]: %s (event_id: %s)",
                error_event.error.type,
                error_event.error.message,
                error_event.event_id,
            )

            # Publish structured error data
            self.event_bus.publish_sync(VoiceAssistantEvent.ERROR_OCCURRED, ErrorEvent)
        except ValidationError as e:
            self.logger.warning("Invalid error event payload: %s", e)
            # Fallback to raw error handling
            error_data = data.get("error", {})
            self.logger.error("OpenAI API error (raw): %s", error_data)
            self.event_bus.publish_sync(
                VoiceAssistantEvent.ERROR_OCCURRED, {"openai_error": error_data}
            )
