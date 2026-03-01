from pydantic import BaseModel

class ApplicationStartedEvent(BaseModel):
    pass

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

class UserStartedSpeakingEvent(BaseModel):
    pass

class UserStoppedSpeakingEvent(BaseModel):
    pass

class AssistantStartedRespondingEvent(BaseModel):
    pass

class AssistantStoppedRespondingEvent(BaseModel):
    pass

class AgentErrorEvent(BaseModel):
    type: str
    message: str
    code: str | None = None
    param: str | None = None

class SubagentCalledEvent(BaseModel):
    agent_name: str
    task: str