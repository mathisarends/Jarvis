from typing import Annotated, Literal

from pydantic import BaseModel, Field

from rtvoice.events.schemas.base import RealtimeServerEvent


class LogProbEntry(BaseModel):
    token: str
    logprob: float
    bytes_: list[int] | None = Field(default=None, alias="bytes")


class TokenInputTokenDetails(BaseModel):
    audio_tokens: int | None = None
    text_tokens: int | None = None


class TokenUsage(BaseModel):
    type: Literal["tokens"]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_token_details: TokenInputTokenDetails | None = None


class DurationUsage(BaseModel):
    type: Literal["duration"]
    seconds: float


Usage = Annotated[TokenUsage | DurationUsage, Field(discriminator="type")]


class InputAudioTranscriptionDelta(BaseModel):
    type: Literal[RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA]
    event_id: str
    item_id: str
    content_index: int
    delta: str
    logprobs: list[LogProbEntry] | None = None


class InputAudioTranscriptionCompleted(BaseModel):
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
    type: Literal[RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA]
    event_id: str
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputAudioTranscriptDone(BaseModel):
    type: Literal[RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE]
    event_id: str
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    transcript: str
