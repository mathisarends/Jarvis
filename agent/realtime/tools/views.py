from __future__ import annotations
from typing import Any, Optional, Literal
from pydantic import BaseModel, ConfigDict, field_validator
import json

from agent.config.views import AgentConfig
from agent.realtime.event_bus import EventBus
from agent.realtime.event_types import RealtimeClientEvent, RealtimeServerEvent
from audio.player.audio_manager import AudioManager


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


class FunctionCallResult(BaseModel):
    tool_name: str
    call_id: str
    output: Optional[Any] = None
    response_instruction: Optional[str] = None

    def to_conversation_item(self) -> dict:
        return {
            "type": RealtimeClientEvent.CONVERSATION_ITEM_CREATE,
            "item": {
                "type": "function_call_output",
                "call_id": self.call_id,
                "output": self._format_output(),
            },
        }

    def _format_output(self) -> str:
        """Private Methode zur Formatierung des Outputs als String."""
        if self.output is None:
            return ""
        if isinstance(self.output, str):
            return self.output
        try:
            return json.dumps(self.output, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(self.output)


# TODO: This has to be encapsulated in a context object (and be more picky about what goes in here - the ToolExecutro needs all this state)
class SpecialToolParameters(BaseModel):
    """Model defining all special parameters that can be injected into actions"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_manager: AudioManager
    event_bus: EventBus
    agent_config: AgentConfig
