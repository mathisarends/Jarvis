from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.realtime.events.client.session_update import (
    InputAudioNoiseReductionConfig,
    RealtimeModel,
)
from agent.realtime.views import (
    AssistantVoice,
)
from audio.player.audio_strategy import AudioStrategy
from audio.wake_word_listener import PorcupineBuiltinKeyword


class AgentConfig(BaseModel):
    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    instructions: str | None = None
    temperature: float = 0.8
    tool_calling_model_name: str | None = None


class WakeWordConfig(BaseModel):
    keyword: PorcupineBuiltinKeyword = PorcupineBuiltinKeyword.PICOVOICE
    sensitivity: float = Field(0.7, ge=0.0, le=1.0)


class AssistantAudioConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_audio_noise_reduction_config: InputAudioNoiseReductionConfig | None
    voice: AssistantVoice = AssistantVoice.MARIN

    playback_speed: float = 1.0
    audio_playback_strategy: AudioStrategy | None = None
