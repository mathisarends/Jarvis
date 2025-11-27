from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agent.realtime.event_types import RealtimeClientEvent


class InputAudioBufferAppendEvent(BaseModel):
    """Model for 'input_audio_buffer' event (base64-encoded audio chunk)."""

    type: Literal[RealtimeClientEvent.INPUT_AUDIO_BUFFER_APPEND] = Field(
        default=RealtimeClientEvent.INPUT_AUDIO_BUFFER_APPEND
    )
    event_id: str | None = None
    audio: str  # base64-encoded audio bytes

    @classmethod
    def from_audio(cls, audio_base64: str) -> InputAudioBufferAppendEvent:
        """Factory to create an InputAudioBufferAppendEvent from base64 audio string."""
        return cls(
            audio=audio_base64,
        )
