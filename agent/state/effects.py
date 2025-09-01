from abc import ABC
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from audio import SoundFile

if TYPE_CHECKING:
    from agent.state.base import AssistantState


class Activity(StrEnum):
    WAKE_WORD = "wake_word"
    ASR_ONCE = "asr_once"
    RESPOND = "respond"
    SESSION_TIMEOUT = "session_timeout"
    ERROR_RECOVER = "error_recover"


class Effect(ABC):
    """Base class for all state-machine effects.

    Effects are *intentions* that the controller executes.
    They carry no behavior themselves, only data.
    """

    pass


@dataclass
class PlaySound(Effect):
    name: SoundFile


@dataclass
class StartActivity(Effect):
    name: str  # "wake_word", "asr_once", "respond", "session_timeout"


@dataclass
class CancelActivity(Effect):
    name: str


@dataclass
class SetSession(Effect):
    active: bool


@dataclass
class TransitionTo(Effect):
    state: AssistantState
