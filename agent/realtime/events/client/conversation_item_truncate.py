from typing import Literal, Optional
from pydantic import BaseModel, Field

from agent.realtime.event_types import RealtimeClientEvent


class ConversationItemTruncateEvent(BaseModel):
    """see https://platform.openai.com/docs/api-reference/realtime_client_events/conversation/item/truncate"""

    event_id: Optional[str] = None
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE] = Field(
        default=RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE
    )
    item_id: str
    content_index: int = 0
    audio_end_ms: int