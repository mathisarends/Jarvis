import asyncio
import logging
import threading
from importlib.resources import files

import sounddevice as sd
import soundfile as sf

from jarvis.events import EventBus
from jarvis.events.views import (
    ApplicationStartedEvent,
    AgentStartedEvent,
    WakeWordDetectedEvent,
    AgentStoppedEvent,
    AgentErrorEvent,
    UserStartedSpeakingEvent,
    UserInactivityCountdownEvent,
)

logger = logging.getLogger(__name__)


class SoundEffectWatchdog:
    _WAKE_SOUND = str(files("jarvis.sounds").joinpath("wakesound.mp3"))
    _VOICE_ASSISTANT_STARTED = str(files("jarvis.sounds").joinpath("voice_assistant_started.mp3"))
    _STOPPED_SOUND = str(files("jarvis.sounds").joinpath("agent_stopped.mp3"))
    _ERROR_SOUND = str(files("jarvis.sounds").joinpath("error_sound.mp3"))
    _STARTUP_SOUND = str(files("jarvis.sounds").joinpath("startup.mp3"))
    _SESSION_TIMEOUT_SOUND = str(files("jarvis.sounds").joinpath("session_timeout.mp3"))

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(ApplicationStartedEvent, self._on_application_started)
        self._event_bus.subscribe(WakeWordDetectedEvent, self._on_wake_word_detected)
        self._event_bus.subscribe(AgentStartedEvent, self._on_agent_started)
        self._event_bus.subscribe(AgentStoppedEvent, self._on_agent_stopped)
        self._event_bus.subscribe(AgentErrorEvent, self._on_agent_error)
        self._event_bus.subscribe(
            UserInactivityCountdownEvent, self._on_user_inactivity_countdown
        )
        self._event_bus.subscribe(UserStartedSpeakingEvent, self._on_user_started_speaking)

        self._wake_data, self._samplerate = sf.read(self._WAKE_SOUND, dtype="float32")
        self._stopped_data, _ = sf.read(self._STOPPED_SOUND, dtype="float32")
        self._error_data, _ = sf.read(self._ERROR_SOUND, dtype="float32")
        self._voice_assistant_started_data, _ = sf.read(self._VOICE_ASSISTANT_STARTED, dtype="float32")
        self._application_started_data, _ = sf.read(self._STARTUP_SOUND, dtype="float32")
        self._session_timeout_data, _ = sf.read(
            self._SESSION_TIMEOUT_SOUND, dtype="float32"
        )

        self._playback_lock = threading.Lock()
        self._session_timeout_task: asyncio.Task | None = None

    async def _on_application_started(self, _: ApplicationStartedEvent) -> None:
        logger.info("ApplicationStartedEvent received – playing startup sound")
        asyncio.create_task(self._play(self._application_started_data))

    async def _on_wake_word_detected(self, _: WakeWordDetectedEvent) -> None:
        logger.info("WakeWordDetectedEvent received – playing wake sound")
        asyncio.create_task(self._play(self._wake_data))

    async def _on_agent_started(self, _: AgentStartedEvent) -> None:
        logger.info("AgentStartedEvent received – playing voice assistant started sound")
        asyncio.create_task(self._play(self._voice_assistant_started_data))

    async def _on_agent_stopped(self, _: AgentStoppedEvent) -> None:
        self._stop_session_timeout_sound()
        logger.info("AgentStoppedEvent received – playing stopped sound")
        asyncio.create_task(self._play(self._stopped_data))

    async def _on_agent_error(self, event: AgentErrorEvent) -> None:
        logger.warning(
            "AgentErrorEvent received – type=%s message=%s", event.type, event.message
        )
        """ asyncio.create_task(self._play(self._error_data)) """

    async def _on_user_inactivity_countdown(
        self, _: UserInactivityCountdownEvent
    ) -> None:
        if self._session_timeout_task and not self._session_timeout_task.done():
            return

        logger.info(
            "UserInactivityCountdownEvent received – playing session timeout sound"
        )
        self._session_timeout_task = asyncio.create_task(
            self._play(self._session_timeout_data)
        )

    async def _on_user_started_speaking(self, _: UserStartedSpeakingEvent) -> None:
        logger.info("UserStartedSpeakingEvent received – stopping session timeout sound")
        self._stop_session_timeout_sound()

    def _stop_session_timeout_sound(self) -> None:
        if self._session_timeout_task and not self._session_timeout_task.done():
            self._session_timeout_task.cancel()
            sd.stop()
        self._session_timeout_task = None

    async def _play(self, data) -> None:
        logger.debug("Playing sound effect (%d frames)", len(data))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._play_blocking, data)
        logger.debug("Sound effect playback finished")

    def _play_blocking(self, data) -> None:
        with self._playback_lock:
            sd.stop()
            sd.play(data, self._samplerate, blocking=True)