from enum import StrEnum


class RealtimeClientEvent(StrEnum):
    # session
    SESSION_UPDATE = "session.update"

    # input audio buffer
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"

    # conversation item
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_RETRIEVE = "conversation.item.retrieve"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"

    # response
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"

    # transcription session
    TRANSCRIPTION_SESSION_UPDATE = "transcription_session.update"

    # output audio buffer
    OUTPUT_AUDIO_BUFFER_CLEAR = "output_audio_buffer.clear"


class RealtimeServerEvent(StrEnum):
    """Server-to-Client Events for OpenAI Realtime API"""

    # Session / Transcription session
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"

    TRANSCRIPTION_SESSION_CREATED = "transcription_session.created"
    TRANSCRIPTION_SESSION_UPDATED = "transcription_session.updated"

    # Conversation
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_DELETED = "conversation.deleted"

    # Conversation item (lifecycle)
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_ADDED = "conversation.item.added"
    CONVERSATION_ITEM_DONE = "conversation.item.done"
    CONVERSATION_ITEM_RETRIEVED = "conversation.item.retrieved"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
    CONVERSATION_ITEM_DELETED = "conversation.item.deleted"

    # Input audio transcription (user-side ASR)
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA = (
        "conversation.item.input_audio_transcription.delta"
    )
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = (
        "conversation.item.input_audio_transcription.completed"
    )
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_SEGMENT = (
        "conversation.item.input_audio_transcription.segment"
    )
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED = (
        "conversation.item.input_audio_transcription.failed"
    )

    # Input audio buffer
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    INPUT_AUDIO_BUFFER_TIMEOUT_TRIGGERED = "input_audio_buffer.timeout_triggered"

    # Response (high-level)
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"

    # Response output items & content parts
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"

    # Text stream (assistant output text)
    RESPONSE_OUTPUT_TEXT_DELTA = "response.output_text.delta"
    RESPONSE_OUTPUT_TEXT_DONE = "response.output_text.done"

    # Audio transcript (assistant-side transcript of its own audio)
    RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
    RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"

    # Audio stream (assistant audio bytes)
    RESPONSE_OUTPUT_AUDIO_DELTA = "response.output_audio.delta"
    RESPONSE_OUTPUT_AUDIO_DONE = "response.output_audio.done"

    # Tools: classic function-calling
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"

    # Tools: MCP (Model Context Protocol)
    MCP_CALL_ARGUMENTS_DELTA = "mcp_call_arguments.delta"
    MCP_CALL_ARGUMENTS_DONE = "mcp_call_arguments.done"

    MCP_LIST_TOOLS_IN_PROGRESS = "mcp_list_tools.in_progress"
    MCP_LIST_TOOLS_COMPLETED = "mcp_list_tools.completed"
    MCP_LIST_TOOLS_FAILED = "mcp_list_tools.failed"

    RESPONSE_MCP_CALL_IN_PROGRESS = "response.mcp_call.in_progress"
    RESPONSE_MCP_CALL_COMPLETED = "response.mcp_call.completed"
    RESPONSE_MCP_CALL_FAILED = "response.mcp_call.failed"

    # Rate limits
    RATE_LIMITS_UPDATED = "rate_limits.updated"

    # Errors
    ERROR = "error"
