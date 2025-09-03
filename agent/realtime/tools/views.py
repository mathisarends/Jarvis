from __future__ import annotations
from typing import Any, Optional, Literal
from pydantic import BaseModel, field_validator
import json

from agent.realtime.event_types import RealtimeServerEvent


class FunctionCallItem(BaseModel):
    """
    One tool/function call request emitted by the model.
    """

    type: Literal[RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE] = (
        RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE
    )

    name: Optional[str] = None
    call_id: str
    event_id: str
    item_id: str
    output_index: int
    response_id: str
    arguments: dict[str, Any]

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> dict[str, Any]:
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            if not v.strip():
                return {}
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # fallback: gib den raw string in einem Feld zurück,
                # damit nichts verloren geht
                return {"__raw__": v}
        raise TypeError("arguments must be a dict or a JSON string")
