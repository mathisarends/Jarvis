from __future__ import annotations
from typing import Any, Callable, Dict
from pydantic import ValidationError

from agent.realtime.event_types import RealtimeServerEvent
from agent.realtime.event_bus import EventBus
from agent.state.base import VoiceAssistantEvent
from agent.realtime.transcription.views import (
    InputAudioTranscriptionCompleted,
    ResponseOutputAudioTranscriptDone,
)
from agent.realtime.views import ResponseOutputAudioDelta
from shared.logging_mixin import LoggingMixin
from shared.singleton_decorator import singleton


@singleton
class RealtimeEventHandler(LoggingMixin):
    """
    Handles OpenAI Realtime API events and maps them to internal VoiceAssistantEvents.
    Provides clean separation between WebSocket message handling and business logic.
    """

    def __init__(self):
        self.event_bus = EventBus()
        
        # Event mapping: OpenAI API event -> handler method
        self._event_handlers: Dict[RealtimeServerEvent, Callable[[Dict[str, Any]], None]] = {
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED: self._handle_user_speech_started,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED: self._handle_user_speech_stopped,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA: self._handle_audio_chunk,
            RealtimeServerEvent.RESPONSE_DONE: self._handle_response_completed,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED: self._handle_user_transcript_completed,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE: self._handle_assistant_transcript_completed,
        }

        # Events we want to log but not handle (for debugging)
        self._ignored_events = {
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

    def handle_openai_event(self, data: dict[str, Any]) -> None:
        """
        Main entry point for handling OpenAI Realtime API events.
        Routes events to appropriate handlers based on event type.
        """
        event_type_str = data.get("type", "")
        
        # Try to convert to enum, fallback to string if not found
        try:
            event_type = RealtimeServerEvent(event_type_str)
        except ValueError:
            # Handle string-only events or unknown events
            if event_type_str in self._ignored_events or event_type in self._ignored_events:
                self.logger.debug("Received ignored event: %s", event_type_str)
                return
            elif event_type_str == "error":
                self._handle_api_error(data)
                return
            else:
                self.logger.warning("Unknown OpenAI event type: %s", event_type_str)
                return

        # Route to appropriate handler
        if event_type in self._ignored_events:
            self.logger.debug("Received ignored event: %s", event_type)
            return
            
        handler = self._event_handlers.get(event_type)
        if handler:
            try:
                handler(data)
            except Exception as e:
                self.logger.error("Error handling event %s: %s", event_type, e)
        else:
            self.logger.debug("No handler registered for event: %s", event_type)

    def _handle_user_speech_started(self, data: Dict[str, Any]) -> None:
        """User started speaking -> USER_STARTED_SPEAKING"""
        self.logger.debug("User started speaking")
        self.event_bus.publish_sync(VoiceAssistantEvent.USER_STARTED_SPEAKING)

    def _handle_user_speech_stopped(self, data: Dict[str, Any]) -> None:
        """User stopped speaking -> USER_SPEECH_ENDED"""
        self.logger.debug("User speech ended")
        self.event_bus.publish_sync(VoiceAssistantEvent.USER_SPEECH_ENDED)

    def _handle_audio_chunk(self, data: Dict[str, Any]) -> None:
        """Audio chunk received -> AUDIO_CHUNK_RECEIVED"""
        try:
            audio_data = ResponseOutputAudioDelta.model_validate(data)
            if not audio_data.delta:
                self.logger.warning("Received empty audio delta")
                return

            self.event_bus.publish_sync(
                VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED, 
                audio_data.delta
            )
        except ValidationError as e:
            self.logger.warning("Invalid audio delta payload: %s", e)

    def _handle_response_completed(self, data: Dict[str, Any]) -> None:
        """Response completed -> ASSISTANT_RESPONSE_COMPLETED"""
        self.logger.debug("Assistant response completed")
        self.event_bus.publish_sync(VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED)

    def _handle_user_transcript_completed(self, data: Dict[str, Any]) -> None:
        """User transcript completed -> USER_TRANSCRIPT_COMPLETED"""
        try:
            payload = InputAudioTranscriptionCompleted.model_validate(data)
            self.event_bus.publish_sync(
                VoiceAssistantEvent.USER_TRANSCRIPT_COMPLETED, 
                payload
            )
        except ValidationError as e:
            self.logger.warning("Invalid user transcript payload: %s", e)

    def _handle_assistant_transcript_completed(self, data: Dict[str, Any]) -> None:
        """Assistant transcript completed -> ASSISTANT_TRANSCRIPT_COMPLETED"""
        try:
            payload = ResponseOutputAudioTranscriptDone.model_validate(data)
            self.event_bus.publish_sync(
                VoiceAssistantEvent.ASSISTANT_TRANSCRIPT_COMPLETED, 
                payload
            )
        except ValidationError as e:
            self.logger.warning("Invalid assistant transcript payload: %s", e)

    def _handle_api_error(self, data: Dict[str, Any]) -> None:
        """API error -> ERROR_OCCURRED"""
        error_data = data.get("error", {})
        self.logger.error("OpenAI API error: %s", error_data)
        self.event_bus.publish_sync(
            VoiceAssistantEvent.ERROR_OCCURRED, 
            {"openai_error": error_data}
        )