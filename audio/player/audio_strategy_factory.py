from enum import StrEnum

from agent.realtime.event_bus import EventBus
from audio.player.audio_strategy import AudioStrategy
from audio.player.pyaudio_strategy import PyAudioStrategy


class AudioStrategyType(StrEnum):
    """Available audio playback strategies"""

    PYAUDIO = "pyaudio"


def create_audio_strategy(
    strategy_type: AudioStrategyType, event_bus: EventBus
) -> AudioStrategy:
    if strategy_type == AudioStrategyType.PYAUDIO:
        return PyAudioStrategy(event_bus=event_bus)
    raise ValueError(f"Unsupported audio strategy: {strategy_type}")
