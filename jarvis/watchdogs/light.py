import asyncio
import logging

from hueify import Hueify, Light

from jarvis.events import EventBus
from jarvis.events.views import (
    WakeWordDetected,
    AgentError,
    AgentInterrupted,
    AgentStopped,
)

logger = logging.getLogger(__name__)


class LightsWatchdog:
    _LIGHT_NAME = "Hue lightstrip plus 1"
    _FLASH_DURATION = 1
    _SHORT_FLASH_DURATION = 0.3

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(WakeWordDetected, self._on_wake_word_detected)
        self._event_bus.subscribe(AgentError, self._on_agent_error)
        self._event_bus.subscribe(AgentInterrupted, self._on_agent_interrupted)
        self._event_bus.subscribe(AgentStopped, self._on_agent_stopped)

        self._light: Light | None = None
        self._hueify: Hueify | None = None

    @property
    def is_connected(self) -> bool:
        return self._light is not None and self._hueify is not None

    @property
    def _is_ready(self) -> bool:
        return self.is_connected and self._light.is_on

    async def start(self) -> None:
        hueify = Hueify()
        await hueify.connect()
        self._light = hueify.lights.from_name(self._LIGHT_NAME)
        self._hueify = hueify
        logger.info("Lights watchdog started")

    async def stop(self) -> None:
        if self.is_connected:
            await self._hueify.close()

    async def _on_wake_word_detected(self, _: WakeWordDetected) -> None:
        if not self._is_ready:
            return
        asyncio.create_task(self._flash())

    async def _flash(self) -> None:
        await self._light.increase_brightness(20)
        await asyncio.sleep(self._FLASH_DURATION)
        await self._light.decrease_brightness(20)

    async def _on_agent_error(self, event: AgentError) -> None:
        logger.warning(
            "AgentError received – type=%s message=%s", event.type, event.message
        )
        if not self._is_ready:
            return
        asyncio.create_task(self._flash_error())

    async def _flash_error(self) -> None:
        for _ in range(3):
            await self._light.increase_brightness(30)
            await asyncio.sleep(self._SHORT_FLASH_DURATION)
            await self._light.decrease_brightness(30)
            await asyncio.sleep(self._SHORT_FLASH_DURATION)

    async def _on_agent_interrupted(self, _: AgentInterrupted) -> None:
        if not self._is_ready:
            return
        asyncio.create_task(self._flash_interrupted())

    async def _flash_interrupted(self) -> None:
        await self._light.decrease_brightness(30)
        await asyncio.sleep(self._SHORT_FLASH_DURATION)
        await self._light.increase_brightness(30)

    async def _on_agent_stopped(self, _: AgentStopped) -> None:
        if not self._is_ready:
            return
        asyncio.create_task(self._flash_stopped())

    async def _flash_stopped(self) -> None:
        await self._light.decrease_brightness(20)
        await asyncio.sleep(self._FLASH_DURATION)
        await self._light.increase_brightness(20)