from abc import ABC, abstractmethod
from agent.realtime.event_bus import EventBus
from agent.state.base import VoiceAssistantEvent
from audio.views import SoundFile
from shared.logging_mixin import LoggingMixin


class AudioStrategy(ABC, LoggingMixin):
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

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Set the event bus for publishing events"""
        self.event_bus = event_bus
        self.logger.info("EventBus has been set")

    def _publish_event(self, event: VoiceAssistantEvent) -> None:
        """Helper method to publish events with warning if no event bus is set"""
        if self.event_bus is None:
            self.logger.warning(
                "Attempted to publish event '%s' but no EventBus is set. "
                "Use set_event_bus() to configure event publishing.",
                event.name if hasattr(event, "name") else str(event),
            )
            return

        try:
            self.event_bus.publish_sync(event)
        except Exception as e:
            self.logger.error("Error publishing event '%s': %s", event, e)

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
