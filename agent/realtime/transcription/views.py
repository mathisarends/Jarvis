from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from agent.realtime.event_types import RealtimeServerEvent


class LogProbEntry(BaseModel):
    """
    A single logprob token with (optional) raw byte values.
    'bytes' is an array of byte values (0..255), as per the documentation.
    """

    token: str
    logprob: float
    bytes_: list[int] | None = Field(default=None, alias="bytes")


class TokenInputTokenDetails(BaseModel):
    audio_tokens: int | None = None
    text_tokens: int | None = None


class TokenUsage(BaseModel):
    # type discriminator
    type: Literal["tokens"]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_token_details: TokenInputTokenDetails | None = None


class DurationUsage(BaseModel):
    # type discriminator
    type: Literal["duration"]
    seconds: float


Usage = Annotated[TokenUsage | DurationUsage, Field(discriminator="type")]


class InputAudioTranscriptionDelta(BaseModel):
    """Delta update for input audio transcription (partial transcript)."""

    type: Literal[RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA]
    event_id: str
    item_id: str
    content_index: int
    delta: str
    logprobs: list[LogProbEntry] | None = None


class InputAudioTranscriptionCompleted(BaseModel):
    """Finalized input audio transcription for a conversation item."""

    type: Literal[
        RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
    ]
    event_id: str
    item_id: str
    content_index: int
    transcript: str
    logprobs: list[LogProbEntry] | None = None
    usage: Usage | None = None


class ResponseOutputAudioTranscriptDelta(BaseModel):
    """Model-generated transcript delta for audio output (streaming)."""

    type: Literal[RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA]
    event_id: str
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputAudioTranscriptDone(BaseModel):
    """Final transcript for audio output when streaming is finished (or interrupted)."""

    type: Literal[RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE]
    event_id: str
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    transcript: str
