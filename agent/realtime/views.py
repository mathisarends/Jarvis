from enum import StrEnum
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

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


class FunctionParameters(BaseModel):
    """JSON Schema for function parameters in a tool."""

    type: str = Field(..., description="Always 'object' for function parameters")
    strict: Optional[bool] = None
    properties: dict[str, Any] = Field(default_factory=dict)
    required: Optional[list[str]] = None


class Tool(BaseModel):
    """A tool available to the model (e.g., function)."""

    type: Literal["function"]
    name: str
    description: Optional[str] = None
    parameters: FunctionParameters


class AudioConfig(BaseModel):
    """Optional audio input/output configuration."""

    input_audio_format: Optional[dict[str, Any]] = None
    output_audio_format: Optional[dict[str, Any]] = None


class SessionConfig(BaseModel):
    """Realtime session configuration for session.update."""

    type: Literal["realtime"]
    model: Optional[str] = None
    audio: Optional[AudioConfig] = None
    include: Optional[list[str]] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[Union[int, Literal["inf"]]] = None
    output_modalities: Optional[list[str]] = None
    prompt: Optional[dict[str, Any]] = None
    temperature: Optional[float] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    tools: Optional[list[Tool]] = None
    tracing: Optional[Union[str, list[str, Any]]] = None
    truncation: Optional[Union[str, list[str, Any]]] = None
