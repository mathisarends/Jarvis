from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, Field

from rtvoice.events.schemas.base import RealtimeClientEvent


class ConversationContent(BaseModel):
    type: Literal["output_text"]
    text: str


class MessageRole(StrEnum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"


class MessageConversationItem(BaseModel):
    type: Literal["message"]
    role: MessageRole
    content: list[ConversationContent]


class FunctionCallOutputConversationItem(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


ConversationItem = MessageConversationItem | FunctionCallOutputConversationItem


class ConversationItemCreateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_CREATE]
    item: ConversationItem

    @classmethod
    def assistant_message(cls, text: str) -> Self:
        return cls(
            type=RealtimeClientEvent.CONVERSATION_ITEM_CREATE,
            item=MessageConversationItem(
                type="message",
                role=MessageRole.ASSISTANT,
                content=[ConversationContent(type="output_text", text=text)],
            ),
        )

    @classmethod
    def function_call_output(cls, call_id: str, output: str) -> Self:
        return cls(
            type=RealtimeClientEvent.CONVERSATION_ITEM_CREATE,
            item=FunctionCallOutputConversationItem(
                type="function_call_output",
                call_id=call_id,
                output=output,
            ),
        )


class ConversationItemTruncateEvent(BaseModel):
    event_id: str | None = None
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE] = Field(
        default=RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE
    )
    item_id: str
    content_index: int = 0
    audio_end_ms: int


class ResponseInstructions(BaseModel):
    instructions: str | None = None


class ConversationResponseCreateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.RESPONSE_CREATE]
    response: ResponseInstructions | None = None

    @classmethod
    def with_instructions(cls, text: str) -> Self:
        return cls(
            type=RealtimeClientEvent.RESPONSE_CREATE,
            response=ResponseInstructions(instructions=text),
        )


class ConversationItemTruncatedEvent(BaseModel):
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE] = Field(
        default=RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE
    )
    event_id: str | None = None
    item_id: str
    content_index: int = 0
    audio_end_ms: int
