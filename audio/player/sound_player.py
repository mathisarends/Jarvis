from typing import Optional
from audio.config import AudioConfig
from audio.player.audio_strategy import AudioStrategy
from audio.views import SoundFile
from audio.player.pyaudio_strategy import PyAudioStrategy
from shared.logging_mixin import LoggingMixin


class SoundPlayer(LoggingMixin):
    """
    Audio player that delegates all functionality to an AudioStrategy implementation.
    Pure delegation pattern - no audio logic here.
    """

    def __init__(
        self, 
        config: Optional[AudioConfig] = None, 
        sounds_dir: Optional[str] = None,
        strategy: Optional[AudioStrategy] = None
    ):
        if strategy is None:
            strategy = PyAudioStrategy(config, sounds_dir)
        
        self.strategy = strategy
        self.logger.info(f"SoundPlayer initialized with {type(strategy).__name__}")

    def set_strategy(self, strategy: AudioStrategy) -> None:
        """Change the audio strategy at runtime"""
        old_strategy_name = type(self.strategy).__name__
        new_strategy_name = type(strategy).__name__
        self.logger.info(f"Switching strategy from {old_strategy_name} to {new_strategy_name}")
        self.strategy = strategy

    def clear_queue_and_stop_chunks(self):
        """Stop current audio playback and clear the audio queue"""
        return self.strategy.clear_queue_and_stop_chunks()

    def is_currently_playing_chunks(self) -> bool:
        """Check if audio is currently playing"""
        return self.strategy.is_currently_playing_chunks()

    def play_sound(self, sound_name: str) -> bool:
        """Play a sound file asynchronously"""
        return self.strategy.play_sound(sound_name)

    def stop_sounds(self):
        """Stop all currently playing sounds"""
        return self.strategy.stop_sounds()

    def get_volume_level(self) -> float:
        """Get the current volume level"""
        return self.strategy.get_volume_level()

    def set_volume_level(self, volume: float) -> float:
        """Set the volume level"""
        return self.strategy.set_volume_level(volume)

    def play_startup_sound(self) -> bool:
        """Play the startup sound"""
        return self.strategy.play_startup_sound()

    def play_wake_word_sound(self) -> bool:
        """Play the wake word sound"""
        return self.strategy.play_wake_word_sound()

    def play_return_to_idle_sound(self) -> bool:
        """Play the return to idle sound"""
        return self.strategy.play_return_to_idle_sound()

    def play_error_sound(self) -> bool:
        """Play the error sound"""
        return self.strategy.play_error_sound()

    def play_sound_file(self, sound_file: SoundFile) -> bool:
        """Play a sound using the SoundFile enum"""
        return self.strategy.play_sound_file(sound_file)

    def add_audio_chunk(self, base64_audio: str):
        """Add a base64 encoded audio chunk to the playback queue"""
        return self.strategy.add_audio_chunk(base64_audio)
