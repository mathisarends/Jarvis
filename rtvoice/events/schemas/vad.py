from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SemanticVadEagerness(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class ServerVadConfig(BaseModel):
    type: Literal["server_vad"]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


class SemanticVadConfig(BaseModel):
    type: Literal["semantic_vad"]
    eagerness: SemanticVadEagerness = SemanticVadEagerness.AUTO
    idle_timeout_ms: int | None = None


class TurnDetectionConfig(BaseModel):
    create_response: bool = True
    interrupt_response: bool = True
    server_vad: ServerVadConfig | None = None
    semantic_vad: SemanticVadConfig | None = None
