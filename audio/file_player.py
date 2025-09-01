import os
from enum import Enum

import pygame
from shared.logging_mixin import LoggingMixin


class SoundFile(Enum):
    """Enum for available sound files."""

    ERROR = "error"
    RETURN_TO_IDLE = "return_to_idle"
    STARTUP = "startup"
    WAKE_WORD = "wake_word"


class SoundFilePlayer(LoggingMixin):
    SUPPORTED_FORMATS = {".mp3"}

    def __init__(self):
        self.volume = 1.0
        self.sounds_dir = os.path.join(os.path.dirname(__file__), "res")

        self.logger.info(
            "Initializing SoundFilePlayer with sounds directory: %s", self.sounds_dir
        )
        self._init_mixer()

    def play_sound(self, sound_name: str) -> bool:
        """Play a sound file asynchronously (non-blocking)."""
        try:
            self._validate_audio_format(sound_name)

            sound_path = self._get_sound_path(sound_name)

            if not os.path.exists(sound_path):
                self.logger.warning("Sound file not found: %s", sound_path)
                return False

            sound = pygame.mixer.Sound(sound_path)
            sound.set_volume(self.volume)
            sound.play()
            self.logger.debug("Playing sound: %s", sound_name)
            return True

        except (RuntimeError, MemoryError, UnicodeDecodeError) as e:
            self.logger.error("Error while playing %s: %s", sound_name, e)
            return False
        except ValueError as e:
            self.logger.error("Format validation failed: %s", e)
            raise  # Re-raise to let caller handle it
        except OSError as e:
            self.logger.error("File access error for %s: %s", sound_name, e)
            return False

    def stop_sound(self) -> None:
        """Stop the currently playing sound."""
        self.logger.info("Stopping current sound playback")

        if not pygame.mixer.get_init():
            self.logger.debug("Pygame mixer not initialized, nothing to stop")
            return

        try:
            pygame.mixer.stop()
            self.logger.info("Pygame mixer stopped")
        except (AttributeError, RuntimeError) as e:
            self.logger.warning("Could not stop pygame mixer: %s", e)

    def play_startup_sound(self) -> bool:
        """Play the startup sound."""
        return self.play_sound_file(SoundFile.STARTUP)

    def play_wake_word_sound(self) -> bool:
        """Play the wake word sound."""
        return self.play_sound_file(SoundFile.WAKE_WORD)

    def play_return_to_idle_sound(self) -> bool:
        """Play the return to idle sound."""
        return self.play_sound_file(SoundFile.RETURN_TO_IDLE)

    def play_error_sound(self) -> bool:
        """Play the error sound."""
        return self.play_sound_file(SoundFile.ERROR)

    def play_sound_file(self, sound_file: SoundFile) -> bool:
        """Play a sound using the SoundFile enum."""
        return self.play_sound(sound_file.value)

    def set_volume_level(self, volume: float) -> None:
        """Set the volume level for the audio player."""
        if not 0.0 <= volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0")

        self.logger.info("Setting volume level to %.2f", volume)
        self.volume = volume

    def get_volume_level(self) -> float:
        """Get the current volume level of the audio player."""
        return self.volume

    def _init_mixer(self):
        """Initialize pygame mixer if not already done."""
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
                self.logger.debug("Pygame mixer initialized in __init__")
            except (RuntimeError, OSError) as e:
                self.logger.error("Failed to init pygame.mixer in __init__: %s", e)
                raise

    def _validate_audio_format(self, sound_name: str) -> None:
        """Validate that the audio format is supported (extension-based check only)."""
        if "." not in sound_name:
            return  # No extension provided, will default to .mp3

        _, ext = os.path.splitext(sound_name.lower())
        if ext in self.SUPPORTED_FORMATS:
            return  # Extension is supported

        # Unsupported extension - raise error
        supported_list = ", ".join(self.SUPPORTED_FORMATS)
        raise ValueError(
            f"Audio format '{ext}' is not supported. "
            f"Supported formats: {supported_list}. "
            f"Please convert '{sound_name}' to MP3 format."
        )

    def _get_sound_path(self, sound_name: str) -> str:
        """Get the full path to a sound file."""
        filename = sound_name if sound_name.endswith(".mp3") else f"{sound_name}.mp3"
        return os.path.join(self.sounds_dir, filename)
