from rtvoice import AgentListener
from rtvoice.views import AgentError

from rtvoice.events.bus import EventBus
from jarvis.events.views import (
    AgentErrorEvent,
    AgentInterruptedEvent,
    AgentStartedEvent,
    AgentStoppedEvent,
    UserStartedSpeakingEvent,
    UserStoppedSpeakingEvent,
    AssistantStartedRespondingEvent,
    AssistantStoppedRespondingEvent,
)


class AgentEventAdapter(AgentListener):
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def on_agent_session_connected(self) -> None:
        await self._event_bus.dispatch(AgentStartedEvent())

    async def on_agent_stopped(self) -> None:
        await self._event_bus.dispatch(AgentStoppedEvent())

    async def on_agent_interrupted(self) -> None:
        await self._event_bus.dispatch(AgentInterruptedEvent())

    async def on_agent_error(self, error: AgentError) -> None:
        await self._event_bus.dispatch(
            AgentErrorEvent(type=error.type, message=error.message, code=error.code, param=error.param)
        )

    async def on_user_started_speaking(self) -> None:
        await self._event_bus.dispatch(UserStartedSpeakingEvent())

    async def on_user_stopped_speaking(self) -> None:
        await self._event_bus.dispatch(UserStoppedSpeakingEvent())

    async def on_assistant_started_responding(self) -> None:
        await self._event_bus.dispatch(AssistantStartedRespondingEvent())

    async def on_assistant_stopped_responding(self) -> None:
        await self._event_bus.dispatch(AssistantStoppedRespondingEvent())