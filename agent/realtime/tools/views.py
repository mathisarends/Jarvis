from typing import Any, Optional, Literal
from pydantic import BaseModel, ConfigDict, field_validator
import json

from agent.realtime.event_bus import EventBus
from agent.realtime.event_types import RealtimeServerEvent
from agent.realtime.events.conversation_item_create import (
    ConversationItemCreateEvent,
    FunctionCallOutputItem,
)
from audio.player.audio_manager import AudioManager
from agent.config.views import AssistantAudioConfig


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

    def to_conversation_item(self) -> ConversationItemCreateEvent:
        return ConversationItemCreateEvent(
            item=FunctionCallOutputItem(
                call_id=self.call_id,
                output=self._format_output(),
            )
        )

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


class SpecialToolParameters(BaseModel):
    """Model defining all special parameters that can be injected into actions"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_manager: AudioManager
    event_bus: EventBus
    assistant_audio_config: AssistantAudioConfig
    tool_calling_model_name: str | None = None

    # optional user-provided context object passed down from Agent(context=...)
    # e.g. can contain anything, external db connections, file handles, queues, runtime config objects, etc.
    # that you might want to be able to access quickly from within many of your actions
    # passed down for convenience, but not automatically used by the system
    context: Any | None = None
