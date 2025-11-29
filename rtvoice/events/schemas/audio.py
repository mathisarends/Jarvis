from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, Field

from rtvoice.events.schemas.base import (
    AssistantVoice,
    RealtimeClientEvent,
    RealtimeServerEvent,
)


class AudioFormat(StrEnum):
    PCM16 = "audio/pcm"
    PCM16U = "audio/pcmu"


class NoiseReductionType(StrEnum):
    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"


class AudioFormatConfig(BaseModel):
    type: AudioFormat = AudioFormat.PCM16
    rate: int = 24000


class AudioOutputConfig(BaseModel):
    voice: AssistantVoice = AssistantVoice.MARIN
    speed: float = Field(default=1.0, ge=0.25, le=1.5)


class InputAudioNoiseReductionConfig(BaseModel):
    type: NoiseReductionType = NoiseReductionType.NEAR_FIELD


class AudioInputConfig(BaseModel):
    format: AudioFormatConfig = Field(
        default_factory=lambda: AudioFormatConfig(type=AudioFormat.PCM16)
    )
    noise_reduction: InputAudioNoiseReductionConfig | None = None


class AudioConfig(BaseModel):
    input: AudioInputConfig = Field(default_factory=AudioInputConfig)
    output: AudioOutputConfig = Field(default_factory=AudioOutputConfig)


class InputAudioBufferAppendEvent(BaseModel):
    type: Literal[RealtimeClientEvent.INPUT_AUDIO_BUFFER_APPEND] = Field(
        default=RealtimeClientEvent.INPUT_AUDIO_BUFFER_APPEND
    )
    event_id: str | None = None
    audio: str

    @classmethod
    def from_audio(cls, audio_base64: str) -> Self:
        return cls(
            audio=audio_base64,
        )


class ResponseOutputAudioDeltaEvent(BaseModel):
    type: Literal[RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA] = (
        RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA
    )
    event_id: str
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    delta: str
