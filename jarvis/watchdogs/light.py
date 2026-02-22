import asyncio
import logging

from hueify import Light

from jarvis.events import EventBus
from jarvis.events.views import WakeWordDetected

logger = logging.getLogger(__name__)

_LIGHT_NAME = "Hue lightstrip plus 1"
_FLASH_DURATION = 0.8


class LightsWatchdog:
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(WakeWordDetected, self._on_wake_word_detected)

    async def _on_wake_word_detected(self, _: WakeWordDetected) -> None:
        asyncio.create_task(self._flash())

    async def _flash(self) -> None:
        light = await Light.from_name(_LIGHT_NAME)
        await light.increase_brightness_percentage(20)
        await asyncio.sleep(_FLASH_DURATION)
        await light.decrease_brightness_percentage(20)