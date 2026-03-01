import asyncio
import logging

from hueify import Hueify, Light

from jarvis.events import EventBus
from jarvis.events.views import (
    WakeWordDetectedEvent,
    AgentStartedEvent,
    AgentErrorEvent,
    AgentInterruptedEvent,
    AgentStoppedEvent,
)

logger = logging.getLogger(__name__)


class LightsWatchdog:
    _LIGHT_NAME = "Hue lightstrip plus 1"
    _FLASH_DURATION = 1
    _SHORT_FLASH_DURATION = 0.3

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(WakeWordDetectedEvent, self._on_wake_word_detected)
        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
        self._event_bus.subscribe(AgentErrorEvent, self._on_agent_error)
        self._event_bus.subscribe(AgentInterruptedEvent, self._on_agent_interrupted)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)

        self._light: Light | None = None
        self._hueify: Hueify | None = None
        self._agent_started_event: asyncio.Event | None = None

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

    async def _on_wake_word_detected(self, _: WakeWordDetectedEvent) -> None:
        if not self._is_ready:
            return
        self._agent_started_event = asyncio.Event()
        asyncio.create_task(self._flash())

    async def _on_agent_started(self, _: AgentStartedEvent) -> None:
        if self._agent_started_event is not None:
            self._agent_started_event.set()

    async def _flash(self) -> None:
        await self._light.increase_brightness(20)
        if self._agent_started_event is not None:
            await self._agent_started_event.wait()
        await self._light.decrease_brightness(20)

    async def _on_agent_error(self, event: AgentErrorEvent) -> None:
        logger.warning(
            "AgentErrorEvent received – type=%s message=%s", event.type, event.message
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

    async def _on_agent_interrupted(self, _: AgentInterruptedEvent) -> None:
        if not self._is_ready:
            return
        asyncio.create_task(self._flash_interrupted())

    async def _flash_interrupted(self) -> None:
        await self._light.decrease_brightness(30)
        await asyncio.sleep(self._SHORT_FLASH_DURATION)
        await self._light.increase_brightness(30)

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        if not self._is_ready:
            return
        asyncio.create_task(self._flash_stopped())

    async def _flash_stopped(self) -> None:
        await self._light.decrease_brightness(20)
        await asyncio.sleep(self._FLASH_DURATION)
        await self._light.increase_brightness(20)