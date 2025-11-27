from __future__ import annotations

from agent.events import EventBus
from agent.state.base import VoiceAssistantEvent
from agent.transcription.models import (
    InputAudioTranscriptionCompleted,
    ResponseOutputAudioTranscriptDone,
)
from shared.logging_mixin import LoggingMixin


class TranscriptionEventListener(LoggingMixin):
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus

        self._setup_event_subscriptions()

    def _setup_event_subscriptions(self) -> None:
        self.event_bus.subscribe(
            VoiceAssistantEvent.USER_TRANSCRIPT_COMPLETED,
            self._handle_user_transcript_completed,
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_TRANSCRIPT_COMPLETED,
            self._handle_assistant_transcript_completed,
        )

    def _handle_user_transcript_completed(
        self, data: InputAudioTranscriptionCompleted
    ) -> None:
        self.logger.info(
            "User transcript completed: '%s' (item_id=%s)",
            data.transcript,
            data.item_id,
        )

        if data.usage:
            self.logger.debug("Transcription usage: %s", data.usage)

    def _handle_assistant_transcript_completed(
        self, data: ResponseOutputAudioTranscriptDone
    ) -> None:
        self.logger.info(
            "Assistant transcript completed: '%s' (response_id=%s)",
            data.transcript,
            data.response_id,
        )
