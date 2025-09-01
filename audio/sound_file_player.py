import os
import pygame

from shared.logging_mixin import LoggingMixin


class SoundFilePlayer(LoggingMixin):
    def __init__(self):
        self.volume = 1.0
        self.sounds_dir = os.path.join(os.path.dirname(__file__), "res")
        
        self.logger.info("Initializing SoundFilePlayer with sounds directory: %s", self.sounds_dir)
        self._init_mixer()

    def _init_mixer(self):
        """Initialize pygame mixer if not already done."""
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
                self.logger.debug("Pygame mixer initialized in __init__")
            except (RuntimeError, OSError) as e:
                self.logger.error("Failed to init pygame.mixer in __init__: %s", e)
                raise  # Oder handle es anders, z.B. mit einem Flag

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

    def play_sound(self, sound_name: str) -> bool:
        """Play a sound file asynchronously (non-blocking)."""
        sound_path = self._get_sound_path(sound_name)

        if not os.path.exists(sound_path):
            self.logger.warning("Sound file not found: %s", sound_path)
            return False

        # Entferne die init-Prüfung hier, da sie bereits in __init__ erfolgt
        try:
            sound = pygame.mixer.Sound(sound_path)
            sound.set_volume(self.volume)
            sound.play()
            self.logger.debug("Playing sound: %s", sound_name)
            return True
        except (RuntimeError, MemoryError, UnicodeDecodeError) as e:
            self.logger.error("Error while playing %s: %s", sound_name, e)
            return False
        except OSError as e:
            self.logger.error("File access error for %s: %s", sound_name, e)
            return False

    def set_volume_level(self, volume: float) -> None:
        """Set the volume level for the audio player."""
        if not 0.0 <= volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0")

        self.logger.info("Setting volume level to %.2f", volume)
        self.volume = volume

    def get_volume_level(self) -> float:
        """Get the current volume level of the audio player."""
        return self.volume

    def _get_sound_path(self, sound_name: str) -> str:
        """Get the full path to a sound file."""
        filename = sound_name if sound_name.endswith(".mp3") else f"{sound_name}.mp3"
        return os.path.join(self.sounds_dir, filename)
    
    
if __name__ == "__main__":
    player = SoundFilePlayer()
    player.play_sound("startup")
    # Optional: Add a delay to let the sound play
    import time
    time.sleep(2)  # Adjust based on sound length
    