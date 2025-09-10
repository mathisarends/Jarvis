from __future__ import annotations
from typing import Any, Literal, Optional

from enum import StrEnum
from pydantic import BaseModel, Field, field_validator, model_validator

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

JsonType = Literal["object", "array", "string", "number", "integer", "boolean"]


class FunctionParameterProperty(BaseModel):
    """
    Property schema for function parameters.
    Minimal: nur was du in deinem Beispiel brauchst.
    """

    type: JsonType
    description: Optional[str] = None


class FunctionParameters(BaseModel):
    type: str = "object"
    strict: bool = True

    properties: dict[str, FunctionParameterProperty] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class FunctionTool(BaseModel):
    """Function tool configuration."""

    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: FunctionParameters


class MCPToolFilter(BaseModel):
    """Filter object for MCP allowed tools."""

    patterns: list[str] | None = None
    exclude: list[str] | None = None


class MCPRequireApproval(BaseModel):
    """Configuration for MCP tool approval requirements."""

    # The documentation doesn't specify the exact structure for object type
    # This is a placeholder that can be extended
    tools: list[str] | None = None
    all: bool | None = None


class MCPTool(BaseModel):
    """MCP tool configuration for Model Context Protocol servers."""

    type: Literal["mcp"] = "mcp"
    server_label: str
    allowed_tools: list[str] | MCPToolFilter | None = None
    authorization: str | None = None
    connector_id: MCPConnectorId | str | None = None
    headers: dict[str, Any] | None = None
    require_approval: MCPRequireApproval | str | None = None
    server_description: str | None = None
    server_url: str | None = None

    @model_validator(mode="after")
    def validate_server_config(self) -> MCPTool:
        """Validate that either server_url or connector_id is provided."""
        if not self.server_url and not self.connector_id:
            raise ValueError("Either 'server_url' or 'connector_id' must be provided")
        return self


class ToolChoice(BaseModel):
    """Tool choice configuration."""

    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    function: FunctionTool | None = None
    mcp: MCPTool | None = None


class MCPConnectorId(StrEnum):
    """Supported MCP connector IDs for service connectors."""

    DROPBOX = "connector_dropbox"
    GMAIL = "connector_gmail"
    GOOGLE_CALENDAR = "connector_googlecalendar"
    GOOGLE_DRIVE = "connector_googledrive"
    MICROSOFT_TEAMS = "connector_microsoftteams"
    OUTLOOK_CALENDAR = "connector_outlookcalendar"
    OUTLOOK_EMAIL = "connector_outlookemail"
    SHAREPOINT = "connector_sharepoint"


class MCPRequireApprovalMode(StrEnum):
    """Approval modes for MCP tool execution"""

    NEVER = "never"
    AUTO = "auto"
    ALWAYS = "always"
    FIRST_USE = "first_use"


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
