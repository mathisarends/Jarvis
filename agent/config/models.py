from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.realtime.events.client.session_update import (
    MCPTool,
    NoiseReductionType,
    RealtimeModel,
    TranscriptionModel,
)
from agent.realtime.views import AssistantVoice
from agent.sound.audio import AudioStrategy, PyAudioStrategy
from agent.wake_word.models import PorcupineWakeWord


class ModelSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    instructions: str | None = None
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    tool_calling_model_name: str | None = None
    mcp_tools: list[MCPTool] | None = None


class VoiceSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    assistant_voice: AssistantVoice = AssistantVoice.MARIN
    speech_speed: float = Field(default=1.0, ge=0.25, le=1.5)
    playback_strategy: AudioStrategy = Field(default_factory=PyAudioStrategy)

    @field_validator("speech_speed")
    @classmethod
    def validate_speech_speed(cls, value: float) -> float:
        if value < 0.25:
            return 0.25
        if value > 1.5:
            return 1.5
        return value


class TranscriptionSettings(BaseModel):
    enabled: bool = False
    model: TranscriptionModel = TranscriptionModel.WHISPER_1
    language: str | None = None
    prompt: str | None = None
    noise_reduction_mode: NoiseReductionType | None = None

    @field_validator("language")
    @classmethod
    def validate_language_code(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("Language code must be a string")

        lang = value.strip().lower()
        if not lang:
            return None

        if len(lang) in (2, 3) and lang.isalpha():
            return lang

        raise ValueError(
            f"Invalid language code: {value!r}. Expected ISO-639-1 format (e.g., 'en', 'de')"
        )


class WakeWordSettings(BaseModel):
    enabled: bool = False
    keyword: PorcupineWakeWord = PorcupineWakeWord.PICOVOICE
    sensitivity: float = Field(default=0.7, ge=0.0, le=1.0)
