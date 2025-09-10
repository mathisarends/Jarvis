from __future__ import annotations

from pydantic import BaseModel, Field

from agent.realtime.events.client.session_update import (
    InputAudioNoiseReductionConfig,
    RealtimeModel,
)
from agent.realtime.views import (
    AssistantVoice,
)
from audio.player.audio_strategy_factory import AudioStrategyType
from audio.wake_word_listener import PorcupineBuiltinKeyword


class AgentConfig(BaseModel):
    """Configuration for the agent"""

    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    instructions: str | None = None
    temperature: float = 0.8
    tool_calling_model_name: str | None = None


class WakeWordConfig(BaseModel):
    """Configuration for the wake word"""

    keyword: PorcupineBuiltinKeyword = PorcupineBuiltinKeyword.PICOVOICE
    sensitivity: float = Field(0.7, ge=0.0, le=1.0)


class AssistantAudioConfig(BaseModel):
    input_audio_noise_reduction_config: InputAudioNoiseReductionConfig | None
    voice: AssistantVoice = AssistantVoice.MARIN

    playback_speed: float = 1.0
    audio_playback_strategy_type: AudioStrategyType | None = None
