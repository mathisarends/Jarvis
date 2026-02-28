import asyncio
import logging
from importlib.resources import files

import sounddevice as sd
import soundfile as sf

from jarvis.events import EventBus
from jarvis.events.views import WakeWordDetected, AgentError, SubagentCalled

logger = logging.getLogger(__name__)


class SoundEffectWatchdog:
    _WAKE_SOUND = str(files("jarvis.sounds").joinpath("wakesound.mp3"))
    _ERROR_SOUND = str(files("jarvis.sounds").joinpath("error_sound.mp3"))
    _SUBAGENT_CALLED_SOUND = str(files("jarvis.sounds").joinpath("subagent_called.mp3"))

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(WakeWordDetected, self._on_wake_word_detected)
        self._event_bus.subscribe(AgentError, self._on_agent_error)
        self._event_bus.subscribe(SubagentCalled, self._on_subagent_called)

        self._wake_data, self._samplerate = sf.read(self._WAKE_SOUND, dtype="float32")
        self._error_data, _ = sf.read(self._ERROR_SOUND, dtype="float32")
        self._hand_off_data, _ = sf.read(self._SUBAGENT_CALLED_SOUND, dtype="float32")

    async def _on_wake_word_detected(self, _: WakeWordDetected) -> None:
        logger.info("WakeWordDetected received – playing wake sound")
        asyncio.create_task(self._play(self._wake_data))

    async def _on_subagent_called(self, event: SubagentCalled) -> None:
        logger.info("SubagentCalled received – agent=%s – playing handoff sound", event.agent_name)
        asyncio.create_task(self._play(self._hand_off_data))

    async def _on_agent_error(self, event: AgentError) -> None:
        logger.warning(
            "AgentError received – type=%s message=%s", event.type, event.message
        )
        asyncio.create_task(self._play(self._error_data))

    async def _play(self, data) -> None:
        logger.debug("Playing sound effect (%d frames)", len(data))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: sd.play(data, self._samplerate, blocking=True))
        logger.debug("Sound effect playback finished")