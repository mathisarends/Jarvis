from pydantic import BaseModel, ConfigDict, Field

from agent.events.schemas import (
    AssistantVoice,
    InputAudioNoiseReductionConfig,
    RealtimeModel,
)
from agent.sound.audio.strategy import AudioStrategy
from agent.wake_word.models import PorcupineWakeWord


class AgentConfig(BaseModel):
    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    instructions: str | None = None
    temperature: float = 0.8
    tool_calling_model_name: str | None = None


class WakeWordConfig(BaseModel):
    keyword: PorcupineWakeWord = PorcupineWakeWord.PICOVOICE
    sensitivity: float = Field(0.7, ge=0.0, le=1.0)


class AssistantAudioConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_audio_noise_reduction_config: InputAudioNoiseReductionConfig | None
    voice: AssistantVoice = AssistantVoice.MARIN

    playback_speed: float = 1.0
    audio_playback_strategy: AudioStrategy | None = None
