
from typing import Literal, Optional

from openai import BaseModel
from agent.realtime.event_types import RealtimeClientEvent

class ResponseInstructions(BaseModel):
    instructions: Optional[str] = None

class ConversationResponseCreateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.RESPONSE_CREATE] = RealtimeClientEvent.RESPONSE_CREATE
    response: ResponseInstructions
    
    