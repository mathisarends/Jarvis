from enum import StrEnum
from dataclasses import dataclass
from typing import Literal, Any
from pydantic import BaseModel, Field

from agent.realtime.event_types import RealtimeClientEvent, RealtimeServerEvent
from audio.wake_word_listener import PorcupineBuiltinKeyword


class RealtimeModel(StrEnum):
    """Available OpenAI Realtime models."""

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


class ErrorDetails(BaseModel):
    """
    Details of an OpenAI Realtime API error.
    Contains information about what went wrong and potentially how to fix it.
    """

    message: str  # Human-readable error message
    type: str  # Error type (e.g., "invalid_request_error", "server_error")
    code: str | None = None  # Error code, if any
    event_id: str | None = None  # Event ID of client event that caused error
    param: str | None = None  # Parameter related to the error


class ErrorEvent(BaseModel):
    """
    OpenAI Realtime API error event.
    Returned when an error occurs, which could be a client problem or server problem.
    Most errors are recoverable and the session will stay open.
    """

    type: Literal[RealtimeServerEvent.ERROR]
    event_id: str
    error: ErrorDetails


# Session Configuration Models


class AudioFormat(StrEnum):
    """Supported audio formats for input/output."""

    PCM16 = "audio/pcm"
    PCM16U = "audio/pcmu"


class NoiseReductionConfig(BaseModel):
    """Configuration for input audio noise reduction."""

    type: str = "auto"  # Currently only "auto" is supported


class TranscriptionConfig(BaseModel):
    """Configuration for input audio transcription."""

    model: str = "whisper-1"
    language: str | None = None
    prompt: str | None = None


class ServerVadConfig(BaseModel):
    """Configuration for Server VAD turn detection."""

    type: Literal["server_vad"]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


class SemanticVadConfig(BaseModel):
    """Configuration for Semantic VAD turn detection."""

    type: Literal["semantic_vad"]
    eagerness: Literal["low", "medium", "high", "auto"] = "auto"
    idle_timeout_ms: int | None = None


class TurnDetectionConfig(BaseModel):
    """Configuration for turn detection."""

    create_response: bool = True
    interrupt_response: bool = True
    # Union of VAD configurations
    server_vad: ServerVadConfig | None = None
    semantic_vad: SemanticVadConfig | None = None


class AudioFormatConfig(BaseModel):
    """Audio format configuration object."""

    type: AudioFormat = AudioFormat.PCM16


class AudioInputConfig(BaseModel):
    """Configuration for input audio."""

    format: AudioFormatConfig = Field(
        default_factory=lambda: AudioFormatConfig(type=AudioFormat.PCM16)
    )
    noise_reduction: NoiseReductionConfig | None = None
    transcription: TranscriptionConfig | None = None
    turn_detection: TurnDetectionConfig | None = None


class AudioOutputConfig(BaseModel):
    """Configuration for output audio."""

    voice: str = "alloy"
    speed: float = Field(default=1.0, ge=0.25, le=1.5)


class AudioConfig(BaseModel):
    """Configuration for input and output audio."""

    input: AudioInputConfig = Field(default_factory=AudioInputConfig)
    output: AudioOutputConfig = Field(default_factory=AudioOutputConfig)


class ClientSecretConfig(BaseModel):
    """Configuration for ephemeral token expiration."""

    expires_after: dict[str, Any] | None = None


class ToolChoiceMode(StrEnum):
    """Tool choice modes."""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


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


class TracingConfig(BaseModel):
    """Configuration for tracing."""

    workflow_name: str | None = None
    group_id: str | None = None
    metadata: dict[str, Any] | None = None


class RetentionRatioTruncation(BaseModel):
    """Retention ratio truncation configuration."""

    type: Literal["retention_ratio"]
    retention_ratio: float = Field(ge=0.0, le=1.0)
    post_instructions_token_limit: int | None = None


class TruncationConfig(BaseModel):
    """Configuration for conversation truncation."""

    type: str = "auto"
    retention_ratio: RetentionRatioTruncation | None = None


class SessionConfig(BaseModel):
    """
    Complete OpenAI Realtime API session configuration.
    Based on the official API documentation.
    """

    type: Literal["realtime"]
    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    instructions: str | None = None
    voice: str | None = None
    audio: AudioConfig = Field(default_factory=AudioConfig)
    client_secret: ClientSecretConfig | None = None
    include: list[str] | None = None
    max_output_tokens: int | str = "inf"  # Can be int or "inf"
    output_modalities: list[str] = ["text", "audio"]
    prompt: dict[str, Any] | None = None
    tool_choice: ToolChoice | ToolChoiceMode = ToolChoiceMode.AUTO
    tools: list[FunctionTool | MCPTool] | None = None
    tracing: TracingConfig | str | None = None  # Can be "auto" or TracingConfig
    truncation: TruncationConfig | str = "auto"  # Can be "auto" or TruncationConfig


class SessionUpdateEvent(BaseModel):
    """
    Event to update the session's default configuration.
    The client may send this at any time to update any field, except for voice.
    """

    type: Literal[RealtimeClientEvent.SESSION_UPDATE]
    event_id: str | None = None
    session: SessionConfig
