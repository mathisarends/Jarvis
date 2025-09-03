from enum import StrEnum
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from agent.realtime.event_types import RealtimeServerEvent
from audio.wake_word_listener import PorcupineBuiltinKeyword


class RealtimeModel(StrEnum):
    GPT_REALTIME = "gpt-realtime"


class AssistantVoice(StrEnum):
    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"
    CEDAR = "cedar"  # only available in gpt-realtime
    MARIN = "marin"  # only available in gpt-realtime


@dataclass(frozen=True)
class VoiceAssistantConfig:
    voice: AssistantVoice = AssistantVoice.ALLOY
    system_message: str = "You are a friendly assistant."
    temperature: float = 0.8

    wake_word: PorcupineBuiltinKeyword = PorcupineBuiltinKeyword.PICOVOICE
    wakeword_sensitivity: float = 0.7
    
    
class ResponseOutputAudioDelta(BaseModel):
    """Model for 'response.output_audio.delta' (base64-encoded audio chunk)."""
    type: Literal[RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA]
    event_id: str
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    delta: str  # base64-encoded audio bytes