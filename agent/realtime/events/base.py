from pydantic import BaseModel, ConfigDict

class RealtimeEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str