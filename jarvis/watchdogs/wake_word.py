import logging
from jarvis.events import EventBus, WakeWordDetected
from jarvis.wake_word import WakeWord, WakeWordListener

logger = logging.getLogger(__name__)


class WakeWordWatchdog:
    def __init__(
        self,
        event_bus: EventBus,
        wake_word: WakeWord = WakeWord.JARVIS,
        wake_word_sensitivity: float = 0.8,
        access_key: str | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._listener = WakeWordListener(
            wake_word=wake_word,
            sensitivity=wake_word_sensitivity,
            access_key=access_key,
            on_detection=self._on_wake_word_detected,
        )

    async def _on_wake_word_detected(self) -> None:
        logger.info("Wake word detected – dispatching event...")
        await self._event_bus.dispatch(WakeWordDetected())

    async def start(self) -> None:
        await self._listener.listen()