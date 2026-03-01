import asyncio
import logging
from importlib.resources import files

import sounddevice as sd
import soundfile as sf

from jarvis.events import EventBus
from jarvis.events.views import AgentStarted, WakeWordDetected, AgentStopped, AgentError, SubagentCalled

logger = logging.getLogger(__name__)


class SoundEffectWatchdog:
    _WAKE_SOUND = str(files("jarvis.sounds").joinpath("wakesound.mp3"))
    _VOICE_ASSISTANT_STARTED = str(files("jarvis.sounds").joinpath("voice_assistant_started.mp3"))
    _STOPPED_SOUND = str(files("jarvis.sounds").joinpath("agent_stopped.mp3"))
    _ERROR_SOUND = str(files("jarvis.sounds").joinpath("error_sound.mp3"))
    _SUBAGENT_SOUND = str(files("jarvis.sounds").joinpath("subagent_called.mp3"))

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(WakeWordDetected, self._on_wake_word_detected)
        self._event_bus.subscribe(AgentStarted, self._on_agent_started)
        self._event_bus.subscribe(AgentStopped, self._on_agent_stopped)
        self._event_bus.subscribe(AgentError, self._on_agent_error)
        self._event_bus.subscribe(SubagentCalled, self._on_subagent_called)

        self._wake_data, self._samplerate = sf.read(self._WAKE_SOUND, dtype="float32")
        self._stopped_data, _ = sf.read(self._STOPPED_SOUND, dtype="float32")
        self._error_data, _ = sf.read(self._ERROR_SOUND, dtype="float32")
        self._subagent_data, _ = sf.read(self._SUBAGENT_SOUND, dtype="float32")
        self._voice_assistant_started_data, _ = sf.read(self._VOICE_ASSISTANT_STARTED, dtype="float32")

    async def _on_wake_word_detected(self, _: WakeWordDetected) -> None:
        logger.info("WakeWordDetected received – playing wake sound")
        asyncio.create_task(self._play(self._wake_data))

    async def _on_agent_started(self, _: AgentStarted) -> None:
        logger.info("AgentStarted received – playing voice assistant started sound")
        asyncio.create_task(self._play(self._voice_assistant_started_data))

    async def _on_agent_stopped(self, _: AgentStopped) -> None:
        logger.info("AgentStopped received – playing stopped sound")
        asyncio.create_task(self._play(self._stopped_data))

    async def _on_agent_error(self, event: AgentError) -> None:
        logger.warning(
            "AgentError received – type=%s message=%s", event.type, event.message
        )
        # most of the time it is not critical if an error occurs (see whether we need audio feedback here)
        """ asyncio.create_task(self._play(self._error_data)) """

    async def _on_subagent_called(self, _: SubagentCalled) -> None:
        logger.info("SubagentCalled received – playing subagent sound")
        asyncio.create_task(self._play(self._subagent_data))

    async def _play(self, data) -> None:
        logger.debug("Playing sound effect (%d frames)", len(data))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: (sd.stop(), sd.play(data, self._samplerate, blocking=True)))
        logger.debug("Sound effect playback finished")
