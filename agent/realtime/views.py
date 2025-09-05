from enum import StrEnum
from typing import Literal, Any, Optional
from pydantic import BaseModel, Field, field_validator

from agent.realtime.event_types import RealtimeClientEvent, RealtimeServerEvent


class RealtimeModel(StrEnum):
    """Available OpenAI Realtime models."""

    GPT_REALTIME = "gpt-realtime"


class AssistantVoice(StrEnum):
    """
    Available assistant voices for the OpenAI Realtime API.

    Each voice has distinct characteristics suited for different use-cases
    such as narration, conversational dialogue, or expressive responses.

    - alloy: Neutral and balanced, clean output suitable for general use.
    - ash: Clear and precise; described as a male baritone with a slightly
      scratchy yet upbeat quality. May have limited performance with accents.
    - ballad: Melodic and gentle; community notes suggest a male-sounding voice.
    - coral: Warm and friendly, good for approachable or empathetic tones.
    - echo: Resonant and deep, strong presence in delivery.
    - fable: Not officially documented; often perceived as narrative-like
      and expressive, fitting for storytelling contexts.
    - onyx: Not officially documented; often perceived as darker, strong,
      and confident in tone.
    - nova: Not officially documented; frequently described as bright,
      youthful, or energetic.
    - sage: Calm and thoughtful, measured pacing and a reflective quality.
    - shimmer: Bright and energetic, dynamic expression with high clarity.
    - verse: Versatile and expressive, adapts well across different contexts.
    - cedar: (Realtime-only) – no official description available yet.
    - marin: (Realtime-only) – no official description available yet.
    """

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

    type: Literal["near_field", "far_field"] = "near_field"


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


class TranscriptionModel(StrEnum):
    """Supported transcription models for input_audio_transcription."""

    WHISPER_1 = "whisper-1"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"

    @field_validator("model", mode="before")
    @classmethod
    def _coerce_model(cls, v: Any) -> Any:
        if v is None or isinstance(v, TranscriptionModel):
            return v
        if isinstance(v, str):
            s = v.strip()
            # Try by name
            try:
                return TranscriptionModel[s.upper().replace("-", "_")]
            except Exception:
                # Fallback by value
                for m in TranscriptionModel:
                    if m.value.lower() == s.lower():
                        return m
        raise ValueError(f"Invalid transcription model: {v!r}")


class InputAudioTranscriptionConfig(BaseModel):
    """Configuration for input audio transcription."""

    model: TranscriptionModel = TranscriptionModel.WHISPER_1
    language: str | None = None  # ISO-639-1 code, e.g. "en" for English
    prompt: str | None = None  # Keywords for whisper, free text for gpt-4o models

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, v: Any) -> TranscriptionModel:
        """Validate and coerce transcription model."""
        if v is None:
            return TranscriptionModel.WHISPER_1
        if isinstance(v, TranscriptionModel):
            return v
        if isinstance(v, str):
            s = v.strip()
            # Try by enum name first
            try:
                return TranscriptionModel[s.upper().replace("-", "_")]
            except KeyError:
                # Try by value
                for model in TranscriptionModel:
                    if model.value.lower() == s.lower():
                        return model
        raise ValueError(f"Invalid transcription model: {v!r}")

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

    type: Literal["near_field", "far_field"] = "near_field"


class SessionConfig(BaseModel):
    """
    Complete OpenAI Realtime API session configuration.
    Based on the official API documentation.
    """

    type: Literal["realtime"] = "realtime"
    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    instructions: str | None = None
    voice: str | None = None
    speed: float = Field(default=1.0, ge=0.25, le=1.5)
    temperature: float = Field(default=0.8, ge=0.6, le=1.2)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    client_secret: ClientSecretConfig | None = None
    include: list[str] | None = None
    max_output_tokens: int | Literal["inf"] = "inf"
    input_audio_transcription: Optional[InputAudioTranscriptionConfig] = (
        None  # field not workign
    )
    input_audio_noise_reduction: Optional[InputAudioNoiseReductionConfig] = None
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

    type: Literal[RealtimeClientEvent.SESSION_UPDATE] = (
        RealtimeClientEvent.SESSION_UPDATE
    )
    event_id: str | None = None
    session: SessionConfig


class SessionCreatedEvent(BaseModel):
    type: Literal[RealtimeServerEvent.SESSION_CREATED]
    event_id: str | None = None
    session: SessionConfig


class ConversationItemTruncateEvent(BaseModel):
    """see https://platform.openai.com/docs/api-reference/realtime_client_events/conversation/item/truncate"""

    event_id: Optional[str] = None
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE] = (
        RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE
    )
    item_id: str
    content_index: int = 0
    audio_end_ms: int


# Server-side event for truncated conversation item (for acknowledgment)
class ConversationItemTruncatedEvent(BaseModel):
    "see https://platform.openai.com/docs/api-reference/realtime_server_events/conversation/item/truncated"

    event_id: str
    type: Literal[RealtimeServerEvent.CONVERSATION_ITEM_TRUNCATED] = (
        RealtimeServerEvent.CONVERSATION_ITEM_TRUNCATED
    )
    item_id: str
    content_index: int
    audio_end_ms: int