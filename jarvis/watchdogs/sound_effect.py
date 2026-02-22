import asyncio
import logging
from importlib.resources import files

import sounddevice as sd
import soundfile as sf

from jarvis.events import EventBus
from jarvis.events.views import WakeWordDetected, AgentStopped

logger = logging.getLogger(__name__)


_WAKE_SOUND = str(files("jarvis.sounds").joinpath("wakesound.wav"))
_STOPPED_SOUND = str(files("jarvis.sounds").joinpath("agent_stopped.wav"))


class SoundEffectWatchdog:
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(WakeWordDetected, self._on_wake_word_detected)
        self._event_bus.subscribe(AgentStopped, self._on_agent_stopped)

        self._wake_data, self._samplerate = sf.read(_WAKE_SOUND, dtype="float32")
        self._stopped_data, _ = sf.read(_STOPPED_SOUND, dtype="float32")

    async def _on_wake_word_detected(self, _: WakeWordDetected) -> None:
        logger.info("WakeWordDetected received – playing wake sound")
        asyncio.create_task(self._play(self._wake_data))

    async def _on_agent_stopped(self, _: AgentStopped) -> None:
        logger.info("AgentStopped received – playing stopped sound")
        asyncio.create_task(self._play(self._stopped_data))

    async def _play(self, data) -> None:
        logger.debug("Playing sound effect (%d frames)", len(data))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: sd.play(data, self._samplerate, blocking=True))
        logger.debug("Sound effect playback finished")