from dataclasses import dataclass

import pyaudio


@dataclass(frozen=True)
class AudioConfig:
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
