import asyncio
from typing import AsyncGenerator
import pyaudio
from audio.config import AudioConfig
from shared.logging_mixin import LoggingMixin


class AudioCapture(LoggingMixin):
    """Class for capturing audio from a microphone using PyAudio with async generator support"""

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_active = False
        self.audio_data = []

        self.logger.info("Initialized with %s", self.config)

    def start_stream(self):
        """Start the microphone stream"""
        if self.stream is not None:
            self.logger.warning("Attempting to start stream but one is already active")
            return

        self.stream = self.p.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            input=True,
            frames_per_buffer=self.config.chunk_size,
        )
        self.is_active = True
        self.audio_data = []
        self.logger.info("Microphone stream started")

    async def stream_chunks(
        self, sleep_interval: float = 0.01
    ) -> AsyncGenerator[bytes, None]:
        """
        Async generator that yields audio chunks continuously
        """
        while self.is_active:
            chunk = self._read_chunk()
            if chunk is not None:
                yield chunk
            else:
                # Short sleep to prevent busy waiting when no data is available
                await asyncio.sleep(sleep_interval)

    def cleanup(self) -> None:
        """Free resources"""
        self._stop_stream()
        self.p.terminate()

    def _read_chunk(self) -> bytes | None:
        """Read a chunk of audio data from the microphone"""
        if self.stream and self.is_active:
            try:
                data = self.stream.read(
                    self.config.chunk_size, exception_on_overflow=False
                )
                self.audio_data.append(data)
                return data
            except Exception as e:
                self.logger.warning("Error reading audio chunk: %s", e)
                return None
        return None

    def _stop_stream(self):
        """Stop the microphone stream"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.is_active = False
        self.logger.info("Microphone stream stopped")
