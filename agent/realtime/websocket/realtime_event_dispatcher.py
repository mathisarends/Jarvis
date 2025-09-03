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
        self.event_handlers: dict[str, Callable[[dict[str, Any]], None]] = {
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
            "session.created",
            "session.updated",
            "conversation.created",
            "conversation.item.created",
            "input_audio_buffer.committed",
            "input_audio_buffer.cleared",
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.failed",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.text.delta",
            "response.text.done",
            "response.audio.delta",
            "response.audio.done",
            "response.audio_transcript.delta",
            "response.audio_transcript.done",
        }

    def dispatch_event(self, data: dict[str, Any]) -> None:
        """
        Hauptmethode für das Dispatching von OpenAI Realtime API Events.
        Routet Events basierend auf dem event type zu den entsprechenden Handlern.
        """
        event_type = data.get("type", "")

        if not event_type:
            self.logger.warning("Received event without type field: %s", data)
            return

        # Prüfe ob wir einen Handler für dieses Event haben
        handler = self.event_handlers.get(event_type)
        if handler:
            try:
                handler(data)
            except Exception as e:
                self.logger.error("Error handling event %s: %s", event_type, e)

        elif event_type in self.ignored_events:
            self.logger.debug("Received ignored event: %s", event_type)

        else:
            self.logger.warning("Unknown OpenAI event type: %s", event_type)

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
