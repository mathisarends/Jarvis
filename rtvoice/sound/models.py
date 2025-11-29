from dataclasses import dataclass
from enum import StrEnum

import pyaudio


class SoundFile(StrEnum):
    ERROR = "error"
    RETURN_TO_IDLE = "return_to_idle"
    STARTUP = "startup"
    WAKE_WORD = "wake_word"


@dataclass(frozen=True)
class AudioConfig:
    chunk_size: int = 4096
    format: int = pyaudio.paInt16
    channels: int = 1
    sample_rate: int = 24000
