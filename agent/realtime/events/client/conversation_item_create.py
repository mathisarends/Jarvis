from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from agent.realtime.event_types import RealtimeClientEvent


class ConversationContent(BaseModel):
    type: Literal["output_text"]
    text: str


class ConversationItem(BaseModel):
    type: Literal["message"]
    role: Literal["assistant", "user", "system"]
    content: list[ConversationContent]


class ConversationItemCreateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_CREATE]
    item: ConversationItem

    @classmethod
    def assistant_message(cls, text: str) -> ConversationItemCreateEvent:
        """
        Factory to create a conversation.item.create event with
        a simple assistant message containing output_text.
        """
        return cls(
            type=RealtimeClientEvent.CONVERSATION_ITEM_CREATE,
            item=ConversationItem(
                type="message",
                role="assistant",
                content=[ConversationContent(type="output_text", text=text)],
            ),
        )
