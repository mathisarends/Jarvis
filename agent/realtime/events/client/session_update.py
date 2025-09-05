from typing import Any, Literal, Optional

from enum import StrEnum
from pydantic import BaseModel, Field, field_validator

from agent.realtime.event_types import RealtimeClientEvent


# ============================================================================
# ENUMS - Base definitions
# ============================================================================


class RealtimeModel(StrEnum):
    """Available OpenAI Realtime models."""

    GPT_REALTIME = "gpt-realtime"


class AudioFormat(StrEnum):
    """Supported audio formats for input/output."""

    PCM16 = "audio/pcm"
    PCM16U = "audio/pcmu"


class TranscriptionModel(StrEnum):
    """Supported transcription models for input_audio_transcription."""

    WHISPER_1 = "whisper-1"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"


class NoiseReductionType(StrEnum):
    """Types for input audio noise reduction."""

    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"


class SemanticVadEagerness(StrEnum):
    """Eagerness levels for Semantic VAD."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


class ToolChoiceMode(StrEnum):
    """Tool choice modes."""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


# ============================================================================
# BASIC AUDIO CONFIGURATIONS
# ============================================================================


class AudioFormatConfig(BaseModel):
    """Audio format configuration object."""

    type: AudioFormat = AudioFormat.PCM16
    rate: int = 24000


class AudioOutputConfig(BaseModel):
    """Configuration for output audio."""

    voice: str = "alloy"
    speed: float = Field(default=1.0, ge=0.25, le=1.5)


class InputAudioTranscriptionConfig(BaseModel):
    """Configuration for input audio transcription."""

    model: TranscriptionModel = TranscriptionModel.WHISPER_1
    language: str | None = None
    prompt: str | None = None

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, v: Any) -> str | None:
        """Validate language code format."""
        if v is None or v == "":
            return None
        if isinstance(v, str):
            lang = v.strip().lower()
            # Basic validation for ISO-639-1 format (2-letter codes)
            if len(lang) == 2 and lang.isalpha():
                return lang
            # Allow some common 3-letter codes too
            if len(lang) == 3 and lang.isalpha():
                return lang
        raise ValueError(
            f"Invalid language code: {v!r}. Expected ISO-639-1 format (e.g., 'en', 'de')"
        )


class InputAudioNoiseReductionConfig(BaseModel):
    """Configuration for input audio noise reduction."""

    type: NoiseReductionType = NoiseReductionType.NEAR_FIELD


# ============================================================================
# VAD (Voice Activity Detection) CONFIGURATIONS
# ============================================================================


class ServerVadConfig(BaseModel):
    """Configuration for Server VAD turn detection."""

    type: Literal["server_vad"]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


class SemanticVadConfig(BaseModel):
    """Configuration for Semantic VAD turn detection."""

    type: Literal["semantic_vad"]
    eagerness: SemanticVadEagerness = SemanticVadEagerness.AUTO
    idle_timeout_ms: int | None = None


class TurnDetectionConfig(BaseModel):
    """Configuration for turn detection."""

    create_response: bool = True
    interrupt_response: bool = True
    # Union of VAD configurations
    server_vad: ServerVadConfig | None = None
    semantic_vad: SemanticVadConfig | None = None


# ============================================================================
# COMPOSITE AUDIO CONFIGURATIONS
# ============================================================================


class AudioInputConfig(BaseModel):
    """Configuration for input audio."""

    format: AudioFormatConfig = Field(
        default_factory=lambda: AudioFormatConfig(type=AudioFormat.PCM16)
    )
    noise_reduction: InputAudioNoiseReductionConfig | None = None
    transcription: InputAudioTranscriptionConfig | None = None
    turn_detection: TurnDetectionConfig | None = None


class AudioConfig(BaseModel):
    """Configuration for input and output audio."""

    input: AudioInputConfig = Field(default_factory=AudioInputConfig)
    output: AudioOutputConfig = Field(default_factory=AudioOutputConfig)


# ============================================================================
# TOOL CONFIGURATIONS
# ============================================================================


class FunctionTool(BaseModel):
    """Function tool configuration."""

    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class MCPTool(BaseModel):
    """MCP tool configuration."""

    type: Literal["mcp"]
    # MCP specific fields would go here


class ToolChoice(BaseModel):
    """Tool choice configuration."""

    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    function: FunctionTool | None = None
    mcp: MCPTool | None = None


# ============================================================================
# SESSION CONFIGURATION
# ============================================================================


class RealtimeSessionConfig(BaseModel):
    """
    Complete OpenAI Realtime API session configuration.
    Based on the official API documentation.
    """

    type: Literal["realtime"] = "realtime"
    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    instructions: str | None = None
    voice: str | None = None
    audio: AudioConfig = Field(default_factory=AudioConfig)
    include: list[str] | None = None
    max_output_tokens: int | Literal["inf"] = "inf"
    input_audio_noise_reduction: Optional[InputAudioNoiseReductionConfig] = None
    output_modalities: list[str] = ["text", "audio"]
    tool_choice: ToolChoice | ToolChoiceMode = ToolChoiceMode.AUTO
    tools: list[FunctionTool | MCPTool] | None = None


# ============================================================================
# EVENTS
# ============================================================================


class SessionUpdateEvent(BaseModel):
    """
    Event to update the session's default configuration.
    The client may send this at any time to update any field, except for voice.
    """

    type: Literal[RealtimeClientEvent.SESSION_UPDATE] = (
        RealtimeClientEvent.SESSION_UPDATE
    )
    event_id: str | None = None
    session: RealtimeSessionConfig
