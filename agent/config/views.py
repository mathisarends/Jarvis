from __future__ import annotations

from typing import Literal, Optional, Any

from pydantic import BaseModel, Field, field_validator

from agent.realtime.views import (
    AssistantVoice,
    InputAudioNoiseReductionConfig,
    InputAudioTranscriptionConfig,
    RealtimeModel,
)
from audio.wake_word_listener import PorcupineBuiltinKeyword


class TurnDetectionConfig(BaseModel):
    """Configuration for turn detection"""

    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    type: str = "server_vad"


class AgentConfig(BaseModel):
    """Configuration for the agent"""

    model: Optional[RealtimeModel] = RealtimeModel.GPT_REALTIME
    voice: AssistantVoice = AssistantVoice.MARIN
    instructions: Optional[str] = None
    max_response_output_tokens: int | Literal["inf"] = "inf"
    temperature: float = 0.8
    speed: float = 1.0
    turn_detection: Optional[TurnDetectionConfig] = None
    transcription: Optional[InputAudioTranscriptionConfig] = None
    input_audio_noise_reduction: Optional[InputAudioNoiseReductionConfig] = None

    @field_validator("voice", mode="before")
    @classmethod
    def _coerce_voice(cls, v: Any) -> Any:
        if isinstance(v, AssistantVoice) or v is None:
            return v
        if isinstance(v, str):
            name = v.strip().upper()
            try:
                return AssistantVoice[name]
            except Exception:
                for member in AssistantVoice:
                    if member.value.lower() == v.strip().lower():
                        return member
        raise ValueError(f"Invalid voice: {v!r}")

    @field_validator("max_response_output_tokens", mode="before")
    @classmethod
    def validate_max_response_output_tokens(cls, v: Any) -> int | Literal["inf"]:
        """
        Validate max_response_output_tokens according to OpenAI Realtime API spec.
        Must be an integer between 1 and 4096, or "inf" for maximum.
        """
        if v == "inf" or v == float("inf"):
            return "inf"

        # Handle integer cases
        if isinstance(v, (int, float)):
            # Convert float to int if it's a whole number
            if isinstance(v, float):
                if v.is_integer():
                    v = int(v)
                else:
                    raise ValueError(
                        f"max_response_output_tokens must be an integer between 1 and 4096, or 'inf'. Got: {v!r}"
                    )

            # Validate range
            if not (1 <= v <= 4096):
                raise ValueError(
                    f"max_response_output_tokens must be between 1 and 4096 (inclusive). Got: {v}"
                )
            return v

        # Handle string representations of numbers
        if isinstance(v, str):
            v_stripped = v.strip()

            # Check for "inf" again (case-insensitive)
            if v_stripped.lower() == "inf":
                return "inf"

            # Try to parse as integer
            try:
                parsed = int(v_stripped)
                if not (1 <= parsed <= 4096):
                    raise ValueError(
                        f"max_response_output_tokens must be between 1 and 4096 (inclusive). Got: {parsed}"
                    )
                return parsed
            except ValueError:
                pass  # Fall through to error below

        raise ValueError(
            f"max_response_output_tokens must be an integer between 1 and 4096, or 'inf'. Got: {v!r}"
        )


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


class VoiceAssistantConfig(BaseModel):
    """Main application configuration"""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
