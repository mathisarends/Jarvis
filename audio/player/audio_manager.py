from typing import Optional
from agent.realtime.event_bus import EventBus
from audio.player.audio_strategy import AudioStrategy
from audio.player.pyaudio_strategy import PyAudioStrategy
from shared.logging_mixin import LoggingMixin


class AudioManager(LoggingMixin):
    """Context that holds and switches audio strategies"""

    def __init__(self, event_bus: EventBus, strategy: Optional[AudioStrategy] = None):
        if strategy is None:
            strategy = PyAudioStrategy(event_bus=event_bus)
        self._strategy = strategy

    def set_strategy(self, strategy: AudioStrategy) -> None:
        """Switch audio strategy at runtime"""
        old_name = type(self._strategy).__name__
        new_name = type(strategy).__name__
        self.logger.info(f"Switching from {old_name} to {new_name}")
        self._strategy = strategy

    @property
    def strategy(self) -> AudioStrategy:
        """Get current strategy for direct access (readonly)"""
        return self._strategy
