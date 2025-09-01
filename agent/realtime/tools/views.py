from __future__ import annotations
from typing import Any, Optional, Literal
from pydantic import BaseModel, field_validator


class FunctionCallItem(BaseModel):
    """
    One tool/function call request emitted by the model.
    """

    type: Literal["function_call"] = "function_call"
    name: Optional[str] = None
    call_id: str
    arguments: dict[str, Any]

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> dict[str, Any]:
        # accept already-parsed dicts OR JSON strings
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            import json

            return json.loads(v or "{}")
        raise TypeError("arguments must be a dict or a JSON string")


class FunctionCallBatch(BaseModel):
    tool_calls: list[FunctionCallItem]


class DoneResponseWithToolCall(BaseModel):
    """
    Represents a completed response that includes a tool call result.
    """

    type: Literal["response.done"] = "response.done"
    response: dict[str, Any]
