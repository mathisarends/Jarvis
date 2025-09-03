from enum import StrEnum


# TODO: Hier vllt. eine event handler Klass schreiben die von einem Event auf das näcshte mappt
class RealtimeServerEvent(StrEnum):
    """Server-to-Client Events for OpenAI Realtime API (based on actual log data)"""

    # Input Audio Buffer Events
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"

    # Conversation Events
    CONVERSATION_ITEM_ADDED = "conversation.item.added"
    CONVERSATION_ITEM_DONE = "conversation.item.done"

    # Response Events
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"

    # Audio Output Events (streaming)
    RESPONSE_OUTPUT_AUDIO_DELTA = "response.output_audio.delta"
    RESPONSE_OUTPUT_AUDIO_DONE = "response.output_audio.done"
    RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
    RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"

    # Transcription Events
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA = (
        "conversation.item.input_audio_transcription.delta"
    )
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = (
        "conversation.item.input_audio_transcription.completed"
    )

    ERROR = "error"
