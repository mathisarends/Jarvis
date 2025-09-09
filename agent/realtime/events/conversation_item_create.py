from typing import Literal, Optional

from pydantic import BaseModel
from agent.realtime.event_types import RealtimeClientEvent


class FunctionCallOutputItem(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str
    id: str | None = None
    object: Literal["realtime.item"] | None = None
    status: str | None = None


class ConversationItemCreateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_CREATE] = (
        RealtimeClientEvent.CONVERSATION_ITEM_CREATE
    )
    event_id: Optional[str] = None
    item: FunctionCallOutputItem
    previous_item_id: Optional[str] = None
