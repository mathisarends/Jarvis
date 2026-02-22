import asyncio
import logging
from importlib.resources import files

import sounddevice as sd
import soundfile as sf

from jarvis.events import EventBus, WakeWordDetected

logger = logging.getLogger(__name__)


_WAKE_SOUND = str(files("jarvis.sounds").joinpath("wakesound.wav"))


class SoundEffectWatchdog:
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._event_bus.subscribe(WakeWordDetected, self._on_wake_word_detected)

        self._data, self._samplerate = sf.read(_WAKE_SOUND, dtype="float32")
        self._stream = sd.OutputStream(
            samplerate=self._samplerate,
            channels=self._data.ndim,
            dtype=self._data.dtype,
        )
        self._stream.start()

    async def _on_wake_word_detected(self, _: WakeWordDetected) -> None:
        asyncio.create_task(self._play_wake_sound())

    async def _play_wake_sound(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._play)

    def _play(self) -> None:
        self._stream.write(self._data)