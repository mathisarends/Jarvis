from __future__ import annotations
from re import I
from typing import Any, Callable
from pydantic import ValidationError

from agent.realtime.event_types import RealtimeServerEvent
from agent.realtime.event_bus import EventBus
from agent.realtime.tools.views import FunctionCallItem
from agent.state.base import VoiceAssistantEvent
from agent.realtime.transcription.views import (
    InputAudioTranscriptionCompleted,
    ResponseOutputAudioTranscriptDone,
)
from agent.realtime.views import (
    ConversationItemTruncatedEvent,
    ResponseOutputAudioDelta,
    ErrorEvent,
    SessionCreatedEvent,
)
from shared.logging_mixin import LoggingMixin


class RealtimeEventDispatcher(LoggingMixin):
    """
    Dispatcher für OpenAI Realtime API Events.
    Mapt eingehende API Events auf interne VoiceAssistantEvents und published diese über den EventBus.
    Bietet saubere Trennung zwischen WebSocket-Handling und Event-Verarbeitung.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        # Event mapping: OpenAI API event type -> handler method
        self.event_handlers: dict[
            RealtimeServerEvent, Callable[[dict[str, Any]], None]
        ] = {
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED: self._handle_user_speech_started,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED: self._handle_user_speech_stopped,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA: self._handle_audio_chunk_received,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED: self._handle_user_transcript_completed,
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE: self._handle_assistant_transcript_completed,
            RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE: self._handle_response_function_call_completed,
            RealtimeServerEvent.SESSION_CREATED: self._handle_session_created,
            RealtimeServerEvent.SESSION_UPDATED: self._handle_session_updated,
            RealtimeServerEvent.CONVERSATION_ITEM_TRUNCATED: self._handle_speech_interruption,
            # Response events
            RealtimeServerEvent.RESPONSE_CREATED: self._handle_response_created,
            RealtimeServerEvent.RESPONSE_DONE: self._handle_response_done,
            RealtimeServerEvent.ERROR: self._handle_api_error,
        }

        # Events die wir loggen aber nicht weiter verarbeiten
        self.ignored_events = {
            # Session events
            RealtimeServerEvent.TRANSCRIPTION_SESSION_UPDATED,
            # Conversation events
            RealtimeServerEvent.CONVERSATION_CREATED,
            RealtimeServerEvent.CONVERSATION_DELETED,
            RealtimeServerEvent.CONVERSATION_ITEM_CREATED,
            RealtimeServerEvent.CONVERSATION_ITEM_ADDED,
            RealtimeServerEvent.CONVERSATION_ITEM_DONE,
            RealtimeServerEvent.CONVERSATION_ITEM_RETRIEVED,
            RealtimeServerEvent.CONVERSATION_ITEM_DELETED,
            # Input audio transcription events
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_SEGMENT,
            RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED,
            # Input audio buffer events
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_COMMITTED,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_CLEARED,
            RealtimeServerEvent.INPUT_AUDIO_BUFFER_TIMEOUT_TRIGGERED,
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
            # Audio output events
            RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DONE,
            # Function calling events
            RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA,
            # MCP events
            RealtimeServerEvent.MCP_CALL_ARGUMENTS_DONE,
            RealtimeServerEvent.MCP_CALL_ARGUMENTS_DELTA,
            RealtimeServerEvent.MCP_LIST_TOOLS_IN_PROGRESS,
            RealtimeServerEvent.MCP_LIST_TOOLS_COMPLETED,
            RealtimeServerEvent.MCP_LIST_TOOLS_FAILED,
            RealtimeServerEvent.RESPONSE_MCP_CALL_IN_PROGRESS,
            RealtimeServerEvent.RESPONSE_MCP_CALL_COMPLETED,
            RealtimeServerEvent.RESPONSE_MCP_CALL_FAILED,
            # Rate limits
            RealtimeServerEvent.RATE_LIMITS_UPDATED,
            # Error events werden separat behandelt
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

        try:
            event_type_enum = RealtimeServerEvent(event_type_str)
        except ValueError:
            # Handle string-only events or unknown events
            if event_type_str == "error":
                self._handle_api_error(data)
                return
            else:
                self.logger.warning("Unknown OpenAI event type: %s", event_type_str)
                return

        # Route to appropriate handler
        if event_type_enum in self.ignored_events:
            self.logger.debug("Received ignored event: %s", event_type_str)
            return

        handler = self.event_handlers.get(event_type_enum)
        if handler:
            try:
                handler(data)
            except Exception as e:
                self.logger.error("Error handling event %s: %s", event_type_str, e)
        else:
            self.logger.debug("No handler registered for event: %s", event_type_str)

    def _handle_user_speech_started(self, data: dict[str, Any]) -> None:
        """User started speaking -> USER_STARTED_SPEAKING"""
        self.logger.debug("User started speaking")
        self.event_bus.publish_sync(VoiceAssistantEvent.USER_STARTED_SPEAKING)

    def _handle_user_speech_stopped(self, data: dict[str, Any]) -> None:
        """User stopped speaking -> USER_SPEECH_ENDED"""
        self.logger.debug("User speech ended")
        self.event_bus.publish_sync(VoiceAssistantEvent.USER_SPEECH_ENDED)

    def _handle_audio_chunk_received(self, data: dict[str, Any]) -> None:
        """Audio chunk received -> AUDIO_CHUNK_RECEIVED"""
        try:
            audio_data = ResponseOutputAudioDelta.model_validate(data)
            if not audio_data.delta:
                self.logger.warning("Received empty audio delta")
                return

            self.event_bus.publish_sync(
                VoiceAssistantEvent.AUDIO_CHUNK_RECEIVED, audio_data
            )
        except ValidationError as e:
            self.logger.warning("Invalid audio delta payload: %s", e)

    def _handle_user_transcript_completed(self, data: dict[str, Any]) -> None:
        """User transcript completed -> USER_TRANSCRIPT_COMPLETED"""
        try:
            payload = InputAudioTranscriptionCompleted.model_validate(data)
            self.event_bus.publish_sync(
                VoiceAssistantEvent.USER_TRANSCRIPT_COMPLETED, payload
            )
        except ValidationError as e:
            self.logger.warning("Invalid user transcript payload: %s", e)

    def _handle_assistant_transcript_completed(self, data: dict[str, Any]) -> None:
        """Assistant transcript completed -> ASSISTANT_TRANSCRIPT_COMPLETED"""
        try:
            payload = ResponseOutputAudioTranscriptDone.model_validate(data)
            self.event_bus.publish_sync(
                VoiceAssistantEvent.ASSISTANT_TRANSCRIPT_COMPLETED, payload
            )
        except ValidationError as e:
            self.logger.warning("Invalid assistant transcript payload: %s", e)

    def _handle_response_function_call_completed(self, data: dict[str, Any]) -> None:
        """Response function call completed -> ASSISTANT_STARTED_TOOL_CALL"""
        try:
            function_call_item = FunctionCallItem.model_validate(data)
            self.logger.info(
                "Function call initiated - Tool: %s, Args: %s",
                function_call_item.name,
                function_call_item.arguments,
            )
            self.event_bus.publish_sync(
                VoiceAssistantEvent.ASSISTANT_STARTED_TOOL_CALL, function_call_item
            )

        except ValidationError as e:
            self.logger.error("Failed to validate function call item: %s", e)

    def _handle_session_updated(self, data: dict[str, Any]) -> None:
        """Session updated - debug output"""
        # Handle tool updates here
        pass

    def _handle_speech_interruption(self, data: dict[str, Any]) -> None:
        """Handle speech interruption event -> ASSISTANT_SPEECH_INTERRUPTED"""
        truncate_event = ConversationItemTruncatedEvent.model_validate(data)
        self.logger.info(
            "Conversation item truncated - Item ID: %s, Audio End MS: %s",
            truncate_event.item_id,
            truncate_event.audio_end_ms,
        )
        self.event_bus.publish_sync(
            VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED, truncate_event
        )

    def _handle_mcp_call_arguments_done(self, data: dict[str, Any]) -> None:
        """MCP call arguments done -> ASSISTANT_RECEIVED_MCP_TOOL_CALL_RESULT"""
        print("data", data)
        self.logger.debug("MCP call arguments done")
        self.event_bus.publish_sync(VoiceAssistantEvent.ASSISTANT_STARTED_MCP_TOOL_CALL)

    def _handle_response_created(self, data: dict[str, Any]) -> None:
        """Response created -> ASSISTANT_STARTED_RESPONSE"""
        self.logger.debug("Assistant response started")
        self.event_bus.publish_sync(VoiceAssistantEvent.ASSISTANT_STARTED_RESPONSE)

    def _handle_response_done(self, data: dict[str, Any]) -> None:
        """Response done -> ASSISTANT_COMPLETED_RESPONSE"""
        self.logger.debug("Assistant response completed")
        self.event_bus.publish_sync(VoiceAssistantEvent.ASSISTANT_COMPLETED_RESPONSE)

    def _handle_api_error(self, data: dict[str, Any]) -> None:
        """API error -> ERROR_OCCURRED"""
        try:
            error_event = ErrorEvent.model_validate(data)
            self.logger.error(
                "OpenAI API error: %s (type: %s, code: %s)",
                error_event.error.message,
                error_event.error.type,
                error_event.error.code,
            )
            self.event_bus.publish_sync(VoiceAssistantEvent.ERROR_OCCURRED, error_event)
        except ValidationError as e:
            self.logger.error("Failed to validate error event: %s", e)
            error_data = data.get("error", {})
            self.logger.error("OpenAI API error (raw): %s", error_data)
            self.event_bus.publish_sync(VoiceAssistantEvent.ERROR_OCCURRED, error_data)

    def _handle_session_created(self, data: dict[str, Any]) -> None:
        """Session created - log event details"""
        session_event = SessionCreatedEvent.model_validate(data)
        session_config = session_event.session

        self.logger.info("OpenAI Realtime session created successfully")
        self.logger.info("Model: %s", session_config.model)
        self.logger.info("Voice: %s", session_config.audio.output.voice)
        self.logger.info(
            "Tools: %d", len(session_config.tools) if session_config.tools else 0
        )
        self.logger.info(
            "Instructions: %s",
            (
                session_config.instructions[:50] + "..."
                if session_config.instructions and len(session_config.instructions) > 50
                else session_config.instructions or "None"
            ),
        )
        self.logger.info("Audio Input: %s", session_config.audio.input.format.type)
        self.logger.info("Audio Output: %s", session_config.audio.output.voice)
        self.logger.info("Max Tokens: %s", session_config.max_output_tokens)

        self.logger.debug("Full session config: %s", session_config.model_dump())
