from enum import StrEnum
from typing import Literal
from pydantic import BaseModel

from agent.realtime.event_types import RealtimeServerEvent
from agent.realtime.events.client.session_update import RealtimeSessionConfig


# ============================================================================
# UNIQUE ENUMS (not in the other file)
# ============================================================================


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


# ============================================================================
# SERVER EVENTS (unique to this file)
# ============================================================================


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


class SessionCreatedEvent(BaseModel):
    """Server event when session is created."""

    type: Literal[RealtimeServerEvent.SESSION_CREATED]
    event_id: str | None = None
    session: RealtimeSessionConfig


class ConversationItemTruncatedEvent(BaseModel):
    """Server-side event for truncated conversation item (for acknowledgment)."""

    event_id: str
    type: Literal[RealtimeServerEvent.CONVERSATION_ITEM_TRUNCATED] = (
        RealtimeServerEvent.CONVERSATION_ITEM_TRUNCATED
    )
    item_id: str
    content_index: int
    audio_end_ms: int
