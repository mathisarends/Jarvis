from abc import ABC, abstractmethod

from rtvoice.events.bus import EventBus
from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.sound.models import SoundFile
from rtvoice.state.base import VoiceAssistantEvent


class AudioStrategy(ABC, LoggingMixin):
    @abstractmethod
    def clear_queue_and_stop_chunks(self) -> None:
        pass

    @abstractmethod
    def is_currently_playing_chunks(self) -> bool:
        pass

    @abstractmethod
    def play_sound(self, sound_name: str) -> None:
        pass

    @abstractmethod
    def stop_sounds(self) -> None:
        pass

    @abstractmethod
    def get_volume_level(self) -> float:
        pass

    @abstractmethod
    def set_volume_level(self, volume: float) -> None:
        pass

    @abstractmethod
    def play_sound_file(self, sound_file: SoundFile) -> None:
        pass

    @abstractmethod
    def add_audio_chunk(self, base64_audio: str) -> None:
        pass

    def set_event_bus(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self.logger.info("EventBus has been set")

    def _publish_event(self, event: VoiceAssistantEvent) -> None:
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

    def play_startup_sound(self) -> None:
        self.play_sound_file(SoundFile.STARTUP)

    def play_wake_word_sound(self) -> None:
        self.play_sound_file(SoundFile.WAKE_WORD)

    def play_return_to_idle_sound(self) -> None:
        self.play_sound_file(SoundFile.RETURN_TO_IDLE)

    def play_error_sound(self) -> None:
        self.play_sound_file(SoundFile.ERROR)
