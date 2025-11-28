from typing import Literal

from pydantic import BaseModel

from agent.events.schemas.base import RealtimeServerEvent


class _ErrorDetails(BaseModel):
    message: str
    type: str
    code: str | None = None
    event_id: str | None = None
    param: str | None = None


class ErrorEvent(BaseModel):
    type: Literal[RealtimeServerEvent.ERROR]
    event_id: str
    error: _ErrorDetails
