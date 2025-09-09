from __future__ import annotations
from typing import TYPE_CHECKING

from agent.realtime.event_bus import EventBus, on_event_with_data
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from agent.realtime.transcription.views import (
        InputAudioTranscriptionCompleted,
        ResponseOutputAudioTranscriptDone,
    )


class TranscriptionService(LoggingMixin):
    """
    Service that handles completed transcription events from the EventBus.
    Logs only the final user and assistant transcripts when they are completed.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.event_bus.register_handlers(self)

    @on_event_with_data(VoiceAssistantEvent.USER_TRANSCRIPT_COMPLETED)
    def _handle_user_transcript_completed(
        self, data: InputAudioTranscriptionCompleted
    ) -> None:
        """Handle completed user transcript"""
        self.logger.info(
            "User transcript completed: '%s' (item_id=%s)",
            data.transcript,
            data.item_id,
        )

        # Log usage info if available
        if data.usage:
            self.logger.debug("Transcription usage: %s", data.usage)

    @on_event_with_data(VoiceAssistantEvent.ASSISTANT_TRANSCRIPT_COMPLETED)
    def _handle_assistant_transcript_completed(
        self, data: ResponseOutputAudioTranscriptDone
    ) -> None:
        """Handle completed assistant transcript"""
        self.logger.info(
            "Assistant transcript completed: '%s' (response_id=%s)",
            data.transcript,
            data.response_id,
        )
