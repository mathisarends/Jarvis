import asyncio
import functools
import logging
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pyaudio
import openwakeword
from openwakeword.model import Model

from jarvis.wake_word.views import WAKE_WORD_MODEL, WakeWord

logger = logging.getLogger(__name__)

CHUNK = 1280
RATE = 16000
CHANNELS = 1

MODELS_DIR = Path(openwakeword.__file__).parent / "resources" / "models"


class WakeWordListener:
    def __init__(
        self,
        on_detection: Callable[[], Awaitable[None]],
        wake_word: WakeWord = WakeWord.HEY_JARVIS,
        sensitivity: float = 0.5,
    ) -> None:
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("Sensitivity must be between 0.0 and 1.0.")

        self._wake_word = wake_word
        self._sensitivity = sensitivity
        self._on_detection = on_detection
        model_path = MODELS_DIR / WAKE_WORD_MODEL[wake_word]
        self._model = Model(wakeword_model_paths=[str(model_path)])
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            rate=RATE,
            channels=CHANNELS,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=CHUNK,
        )

    async def listen(self) -> None:
        logger.info('Listening for "%s"...', self._wake_word)

        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            loop.add_signal_handler(signal.SIGINT, self._shutdown)
        else:
            signal.signal(signal.SIGINT, lambda *_: self._shutdown())

        while True:
            pcm = await loop.run_in_executor(
                None,
                functools.partial(self._stream.read, CHUNK, exception_on_overflow=False),
            )
            await self._process_audio(pcm)

    async def _process_audio(self, pcm: bytes) -> None:
        audio = np.frombuffer(pcm, dtype=np.int16)
        predictions = self._model.predict(audio)

        if not predictions:
            return

        score = max(predictions.values())

        if score < self._sensitivity:
            return

        logger.info('Wake word detected (score=%.2f) – pausing listener.', score)
        self._stream.stop_stream()

        await self._on_detection()

        self._model.reset()
        self._stream.start_stream()
        logger.info('Listening for "%s"...', self._wake_word)

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()
        sys.exit(0)