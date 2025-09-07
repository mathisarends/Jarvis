from abc import ABC, abstractmethod
from audio.views import SoundFile


class AudioStrategy(ABC):
    """
    Abstract base class for audio playback strategies.
    """

    @abstractmethod
    def clear_queue_and_stop_chunks(self) -> None:
        """Stop current audio playback and clear the audio queue"""
        pass

    @abstractmethod
    def is_currently_playing_chunks(self) -> bool:
        """Check if audio is currently playing"""
        pass

    @abstractmethod
    def play_sound(self, sound_name: str) -> bool:
        """Play a sound file asynchronously"""
        pass

    @abstractmethod
    def stop_sounds(self) -> None:
        """Stop all currently playing sounds"""
        pass

    @abstractmethod
    def get_volume_level(self) -> float:
        """Get the current volume level"""
        pass

    @abstractmethod
    def set_volume_level(self, volume: float) -> float:
        """Set the volume level"""
        pass

    @abstractmethod
    def play_sound_file(self, sound_file: SoundFile) -> bool:
        """Play a sound using the SoundFile enum"""
        pass

    @abstractmethod
    def add_audio_chunk(self, base64_audio: str) -> None:
        """Add a base64 encoded audio chunk to the playback queue"""
        pass

    def play_startup_sound(self) -> bool:
        """Play the startup sound"""
        return self.play_sound_file(SoundFile.STARTUP)

    def play_wake_word_sound(self) -> bool:
        """Play the wake word sound"""
        return self.play_sound_file(SoundFile.WAKE_WORD)

    def play_return_to_idle_sound(self) -> bool:
        """Play the return to idle sound"""
        return self.play_sound_file(SoundFile.RETURN_TO_IDLE)

    def play_error_sound(self) -> bool:
        """Play the error sound"""
        return self.play_sound_file(SoundFile.ERROR)

    def play_sound_check(self) -> bool:
        """Play the sound check sound"""
        return self.play_sound_file(SoundFile.SOUND_CHECK)

    def play_assistant_interrupted_sound(self) -> bool:
        """Play the assistant interrupted sound"""
        return self.play_sound_file(SoundFile.ASSISTANT_INTERRUPTED)
