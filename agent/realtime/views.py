from enum import StrEnum
from dataclasses import dataclass


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


@dataclass(frozen=True)  # frozen will depend on wether the options can change here
class OpenAIRealtimeConfig:
    """
    Configuration object for the OpenAI Realtime API integration.
    Pass this into OpenAIRealtimeAPI instead of separate params.
    """

    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    voice: AssistantVoice = AssistantVoice.ALLOY
    temperature: float = 0.7


# ---- Client -> Server events -----------------------------------------------
class RealtimeClientEvent(StrEnum):
    # Session / configuration
    SESSION_UPDATE = (
        "session.update"  # Update session config (model, voice, modalities, VAD, etc.)
    )

    # Conversation inputs
    CONVERSATION_ITEM_CREATE = (
        "conversation.item.create"  # Add a message/item to the conversation
    )
    CONVERSATION_ITEM_TRUNCATE = (
        "conversation.item.truncate"  # Trim items (interrupt/trim flows)
    )

    # Response control
    RESPONSE_CREATE = "response.create"  # Ask model to generate a response
    RESPONSE_CANCEL = "response.cancel"  # Cancel an in-progress response

    # Low-level audio (WebSocket transport)
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"  # Stream base64 audio chunk
    INPUT_AUDIO_BUFFER_COMMIT = (
        "input_audio_buffer.commit"  # Close current input buffer (when VAD is off)
    )
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"  # Clear buffer for next turn


# ---- Server -> Client events -----------------------------------------------
class RealtimeServerEvent(StrEnum):
    # Session lifecycle
    SESSION_CREATED = "session.created"  # Session is ready
    SESSION_UPDATED = "session.updated"  # Session config was updated

    # Conversation & response lifecycle
    CONVERSATION_ITEM_CREATED = (
        "conversation.item.created"  # An item was added to conversation
    )
    RESPONSE_CREATED = "response.created"  # A new response was created
    RESPONSE_OUTPUT_ITEM_ADDED = (
        "response.output_item.added"  # Output item appended to response
    )
    RESPONSE_OUTPUT_ITEM_CREATED = "response.output_item.created"  # Output item created
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"  # Output item finished
    RESPONSE_CONTENT_PART_ADDED = (
        "response.content_part.added"  # Content part (e.g., text/audio chunk) added
    )
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"  # Content part done
    RESPONSE_DONE = "response.done"  # Whole response is finished

    # Text streaming
    RESPONSE_TEXT_DELTA = "response.text.delta"  # Text token delta
    RESPONSE_TEXT_DONE = "response.text.done"  # Text stream finished

    # Audio output streaming
    RESPONSE_AUDIO_DELTA = "response.audio.delta"  # Base64 audio chunk
    RESPONSE_AUDIO_DONE = "response.audio.done"  # Audio stream finished

    # Audio transcript (model-generated transcript of output audio)
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = (
        "response.audio_transcript.delta"  # Transcript delta
    )
    RESPONSE_AUDIO_TRANSCRIPT_DONE = (
        "response.audio_transcript.done"  # Transcript finished
    )

    # Function calling (arguments streaming)
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = (
        "response.function_call_arguments.delta"  # Args delta (JSON)
    )
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = (
        "response.function_call_arguments.done"  # Args complete
    )

    # Input audio status (WebSocket input buffer lifecycle)
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = (
        "input_audio_buffer.speech_started"  # VAD: speech started
    )
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = (
        "input_audio_buffer.speech_stopped"  # VAD: speech stopped
    )
    INPUT_AUDIO_BUFFER_COMMITTED = (
        "input_audio_buffer.committed"  # Buffer committed (turn boundary)
    )

    # Limits & errors
    RATE_LIMITS_UPDATED = "rate_limits.updated"  # Rate limits / usage updated
    ERROR = "error"
