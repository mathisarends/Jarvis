import asyncio
import logging

from hueify import Hueify

from jarvis.events import EventBus
from jarvis.events.views import WakeWordDetected, AgentError

logger = logging.getLogger(__name__)

_LIGHT_NAME = "Hue lightstrip plus 1"
_FLASH_DURATION = 0.8


class LightsWatchdog:
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(WakeWordDetected, self._on_wake_word_detected)
        self._event_bus.subscribe(AgentError, self._on_agent_error)
        self._hueify: Hueify | None = None

    @property
    def is_conntected(self) -> bool:
        return self._hueify is not None

    async def start(self) -> None:
        self._hueify = Hueify()
        await self._hueify.connect()
        logger.info("Lights watchdog started")

    async def stop(self) -> None:
        if self.is_conntected:
            await self._hueify.close()

    async def _on_wake_word_detected(self, _: WakeWordDetected) -> None:
        asyncio.create_task(self._flash())

    async def _on_agent_error(self, event: AgentError) -> None:
        logger.warning(
            "AgentError received – type=%s message=%s", event.type, event.message
        )

    async def _flash(self) -> None:
        if not self.is_conntected:
            return
        light = self._hueify.lights.from_name(_LIGHT_NAME)
        await light.increase_brightness(20)
        await asyncio.sleep(_FLASH_DURATION)
        await light.decrease_brightness(20)