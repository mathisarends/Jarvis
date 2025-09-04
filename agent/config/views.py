from __future__ import annotations

from typing import Optional, Any

from pydantic import BaseModel, Field, field_validator

from agent.realtime.event_types import RealtimeClientEvent
from agent.realtime.views import AssistantVoice, RealtimeModel
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
    max_output_tokens: Optional[int] = None
    temperature: float = 0.8
    speed: float = 1.0
    turn_detection: Optional[TurnDetectionConfig] = None

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
