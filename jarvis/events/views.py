from pydantic import BaseModel

class WakeWordDetected(BaseModel):
    pass

class AgentStarted(BaseModel):
    pass

class AgentStopped(BaseModel):
    pass

class AgentInterrupted(BaseModel):
    pass

class AgentError(BaseModel):
    type: str
    message: str
    code: str | None = None
    param: str | None = None