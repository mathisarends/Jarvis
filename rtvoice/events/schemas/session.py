from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from rtvoice.events.schemas.audio import (
    AudioConfig,
    InputAudioNoiseReductionConfig,
)
from rtvoice.events.schemas.base import (
    RealtimeClientEvent,
    RealtimeModel,
    RealtimeServerEvent,
)
from rtvoice.tools.models import FunctionTool, MCPTool, ToolChoice, ToolChoiceMode


class OutputModality(StrEnum):
    TEXT = "text"
    AUDIO = "audio"


class RealtimeSessionConfig(BaseModel):
    type: Literal["realtime"] = "realtime"
    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    instructions: str | None = None
    voice: str | None = None
    audio: AudioConfig = Field(default_factory=AudioConfig)
    include: list[str] | None = None
    max_output_tokens: int | Literal["inf"] = "inf"
    input_audio_noise_reduction: InputAudioNoiseReductionConfig | None = None
    output_modalities: list[OutputModality] = Field(
        default_factory=lambda: [OutputModality.AUDIO]
    )
    tool_choice: ToolChoice | ToolChoiceMode = ToolChoiceMode.AUTO
    tools: list[FunctionTool | MCPTool] | None = None


class SessionUpdateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.SESSION_UPDATE] = (
        RealtimeClientEvent.SESSION_UPDATE
    )
    event_id: str | None = None
    session: RealtimeSessionConfig


class SessionCreatedEvent(BaseModel):
    type: Literal[RealtimeServerEvent.SESSION_CREATED] = (
        RealtimeServerEvent.SESSION_CREATED
    )
    event_id: str | None = None
    session: RealtimeSessionConfig
