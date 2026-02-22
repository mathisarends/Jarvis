from pydantic import BaseModel

class WakeWordDetected(BaseModel):
    pass

class AgentStopped(BaseModel):
    pass

class AgentInterrupted(BaseModel):
    pass