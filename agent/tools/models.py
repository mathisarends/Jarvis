import json
from typing import (
    Any,
    Literal,
)

from pydantic import BaseModel, ConfigDict, field_validator

from agent.config.models import VoiceSettings
from agent.events import EventBus
from agent.events.schemas.base import RealtimeServerEvent
from agent.events.schemas.conversation import (
    ConversationItemCreateEvent,
)
from agent.events.schemas.tools import FunctionCallOutputItem
from agent.sound.player import AudioPlayer


class FunctionCallItem(BaseModel):
    type: Literal[RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE] = (
        RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE
    )

    name: str | None = None
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
    output: Any | None = None
    response_instruction: str | None = None

    def to_conversation_item(self) -> ConversationItemCreateEvent:
        return ConversationItemCreateEvent(
            item=FunctionCallOutputItem(
                call_id=self.call_id,
                output=self._format_output(),
            )
        )

    def _format_output(self) -> str:
        if self.output is None:
            return ""
        if isinstance(self.output, str):
            return self.output
        try:
            return json.dumps(self.output, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(self.output)


class SpecialToolParameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_player: AudioPlayer
    event_bus: EventBus
    voice_settings: VoiceSettings
    tool_calling_model_name: str | None = None
