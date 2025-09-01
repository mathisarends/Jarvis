from dataclasses import dataclass

import pyaudio
from shared.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class AudioCaptureConfig:
    """Audio configuration settings"""

    chunk_size: int = 4096
    format: int = pyaudio.paInt16
    channels: int = 1
    sample_rate: int = 24000

    @property
    def bytes_per_sample(self) -> int:
        """Calculate bytes per sample based on format"""
        format_map = {
            pyaudio.paInt16: 2,
            pyaudio.paInt24: 3,
            pyaudio.paInt32: 4,
            pyaudio.paFloat32: 4,
        }
        return format_map.get(self.format, 2)

    @property
    def chunk_bytes(self) -> int:
        """Calculate total bytes per chunk"""
        return self.chunk_size * self.bytes_per_sample * self.channels

    def __str__(self) -> str:
        return (
            f"AudioConfig(rate={self.sample_rate}Hz, "
            f"chunk={self.chunk_size} samples, "
            f"format={self.bytes_per_sample*8}-bit, "
            f"channels={self.channels}, "
            f"chunk_bytes={self.chunk_bytes})"
        )


class AudioCapture(LoggingMixin):
    """Class for capturing audio from a microphone using PyAudio"""

    def __init__(self, config: AudioCaptureConfig = None):
        self.config = config or AudioCaptureConfig()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_active = False
        self.audio_data = []

        self.logger.info("Initialized with %s", self.config)

    def start_stream(self):
        """Start the microphone stream"""
        if self.stream is not None:
            self.stop_stream()

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

    def stop_stream(self):
        """Stop the microphone stream"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.is_active = False
        self.logger.info("Microphone stream stopped")

    def read_chunk(self):
        """Read a chunk of audio data from the microphone"""
        if self.stream and self.is_active:
            data = self.stream.read(self.config.chunk_size, exception_on_overflow=False)
            self.audio_data.append(data)
            return data
        return None

    def cleanup(self):
        """Free resources"""
        self.stop_stream()
        self.p.terminate()
