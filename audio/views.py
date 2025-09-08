from enum import StrEnum


class SoundFile(StrEnum):
    """Enum for available sound files."""

    ERROR = "error"
    RETURN_TO_IDLE = "return_to_idle"
    STARTUP = "startup"
    WAKE_WORD = "wake_word"
