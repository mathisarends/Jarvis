from pydantic import BaseModel

class WakeWordDetected(BaseModel):
    pass

class AgentStarted(BaseModel):
    pass

class AgentStopped(BaseModel):
    pass

class AgentInterrupted(BaseModel):
    pass