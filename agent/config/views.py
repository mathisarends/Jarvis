from __future__ import annotations

from typing import Literal, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.realtime.events.client.session_update import (
    InputAudioNoiseReductionConfig,
    InputAudioTranscriptionConfig,
    NoiseReductionType,
    RealtimeModel,
)
from agent.realtime.views import (
    AssistantVoice,
)
from audio.player.audio_strategy import AudioStrategy
from audio.wake_word_listener import PorcupineBuiltinKeyword


class AgentConfig(BaseModel):
    """Configuration for the agent"""

    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    voice: AssistantVoice = AssistantVoice.MARIN
    speed: float = 1.0
    instructions: str | None = None
    temperature: float = 0.8
    input_audio_noise_reduction: InputAudioNoiseReductionConfig | None = None
    transcription: InputAudioTranscriptionConfig | None = None
    tool_calling_model_name: str | None = None


class WakeWordConfig(BaseModel):
    """Configuration for the wake word"""

    keyword: PorcupineBuiltinKeyword = PorcupineBuiltinKeyword.PICOVOICE
    sensitivity: float = Field(0.7, ge=0.0, le=1.0)

    @field_validator("keyword", mode="before")
    @classmethod
    def _coerce_keyword(cls, v: Any) -> Any:
        if isinstance(v, PorcupineBuiltinKeyword) or v is None:
            return v
        if isinstance(v, str):
            name = v.strip().upper()
            try:
                return PorcupineBuiltinKeyword[name]
            except Exception:
                for member in PorcupineBuiltinKeyword:
                    if member.value.lower() == v.strip().lower():
                        return member
        raise ValueError(f"Invalid wake_word: {v!r}")


# TODO: Use this here later on
class AudioConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_audio_noise_reduction: NoiseReductionType = NoiseReductionType.NEAR_FIELD
    playback_speed: float = 1.0
    playback_strategy: AudioStrategy | None = None
