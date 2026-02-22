import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from jarvis.wake_word.views import WakeWord
import pvporcupine
import pyaudio
import struct
import signal
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class WakeWordListener:
    def __init__(
        self,
        on_detection: Callable[[], Awaitable[None]],
        wake_word: WakeWord = WakeWord.PORCUPINE,
        sensitivity: float = 0.5,
        access_key: str | None = None,
    ) -> None:
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("sensitivity muss zwischen 0.0 und 1.0 liegen.")

        self._wake_word = wake_word
        self._on_detection = on_detection
        self._access_key = access_key or os.getenv("PICO_ACCESS_KEY")

        if not self._access_key:
            raise ValueError("No access key found.")

        self._porcupine = pvporcupine.create(
            access_key=self._access_key,
            keywords=[wake_word],
            sensitivities=[sensitivity],
        )

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            rate=self._porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self._porcupine.frame_length,
        )

    async def listen(self) -> None:
        logger.info('Listening for "%s"...', self._wake_word)

        loop = asyncio.get_running_loop()
        signal.signal(signal.SIGINT, lambda sig, frame: self._shutdown())

        while True:
            pcm = await loop.run_in_executor(
                None,
                lambda: self._stream.read(self._porcupine.frame_length, exception_on_overflow=False),
            )
            pcm_unpacked = struct.unpack_from("h" * self._porcupine.frame_length, pcm)

            if self._porcupine.process(pcm_unpacked) >= 0:
                logger.info('Wake word "%s" detected – pausing listener.', self._wake_word)
                self._stream.stop_stream()

                await self._on_detection()

                self._stream.start_stream()
                logger.info('Listening for "%s"...', self._wake_word)

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()
        self._porcupine.delete()
        sys.exit(0)