from __future__ import annotations

from typing import Literal, Optional

from openai import BaseModel
from agent.realtime.event_types import RealtimeClientEvent


class ResponseInstructions(BaseModel):
    instructions: Optional[str] = None


class ConversationResponseCreateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.RESPONSE_CREATE]
    response: Optional[ResponseInstructions] = None

    @classmethod
    def with_instructions(cls, text: str) -> ConversationResponseCreateEvent:
        """
        Factory to quickly create a response.create event with simple instructions.
        """
        return cls(
            type=RealtimeClientEvent.RESPONSE_CREATE,
            response=ResponseInstructions(instructions=text),
        )
