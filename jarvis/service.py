import logging

from jarvis.events import EventBus
from jarvis.wake_word import WakeWord
from jarvis.watchdogs import (
    SoundEffectWatchdog, 
    LightsWatchdog, 
    WakeWordWatchdog
)

logger = logging.getLogger(__name__)


class Jarvis:
    def __init__(
        self,
        wake_word: WakeWord = WakeWord.JARVIS,
        wake_word_sensitivity: float = 0.8,
        access_key: str | None = None,
    ) -> None:
        self._event_bus = EventBus()

        self._wake_word_watchdog = WakeWordWatchdog(event_bus=self._event_bus, wake_word=wake_word, wake_word_sensitivity=wake_word_sensitivity, access_key=access_key)
        self._sound_effect_watchdog = SoundEffectWatchdog(event_bus=self._event_bus)
        self._lights_watchdog = LightsWatchdog(event_bus=self._event_bus)

    async def run(self) -> None:
        await self._wake_word_watchdog.start()

    async def stop(self) -> None:
        ...