import asyncio
from collections.abc import AsyncGenerator

import pyaudio

from agent.sound import AudioConfig
from shared.logging_mixin import LoggingMixin


class MicrophoneCapture(LoggingMixin):
    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._pyaudio = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None

        self.logger.info("Initialized with %s", self.config)

    def __enter__(self):
        self.start_stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    @property
    def is_active(self) -> bool:
        return self._stream is not None

    def start_stream(self) -> None:
        if self._stream is not None:
            self.logger.warning("Stream already active")
            return

        self._stream = self._pyaudio.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
        )
        self.logger.info("Microphone stream started")

    def stop_stream(self) -> None:
        if self._stream is None:
            return

        self._stream.stop_stream()
        self._stream.close()
        self._stream = None
        self.logger.info("Microphone stream stopped")

    def cleanup(self) -> None:
        self.stop_stream()
        self._pyaudio.terminate()
        self.logger.info("Audio capture cleaned up")

    def read_chunk(self) -> bytes | None:
        if self._stream is None:
            return None

        return self._stream.read(self.config.chunk_size, exception_on_overflow=False)

    async def stream_chunks(
        self, sleep_interval: float = 0.01
    ) -> AsyncGenerator[bytes]:
        while self._stream is not None:
            chunk = self.read_chunk()
            if chunk is not None:
                yield chunk
            else:
                await asyncio.sleep(sleep_interval)
