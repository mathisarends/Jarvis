from pydantic import BaseModel

class WakeWordDetectedEvent(BaseModel):
    pass

class AgentStartedEvent(BaseModel):
    pass

class AgentStoppedEvent(BaseModel):
    pass

class AgentStopCommand(BaseModel):
    pass

class AgentInterruptedEvent(BaseModel):
    pass

class AgentErrorEvent(BaseModel):
    type: str
    message: str
    code: str | None = None
    param: str | None = None

class SubagentCalledEvent(BaseModel):
    agent_name: str
    task: str