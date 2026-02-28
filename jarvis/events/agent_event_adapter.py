from rtvoice import AgentListener

from rtvoice.events.bus import EventBus
from jarvis.events.views import AgentError, AgentInterrupted, AgentStarted, AgentStopped, SubagentCalled


class AgentEventAdapter(AgentListener):
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def on_agent_started(self) -> None:
        await self._event_bus.dispatch(AgentStarted())

    async def on_agent_stopped(self) -> None:
        await self._event_bus.dispatch(AgentStopped())

    async def on_agent_interrupted(self) -> None:
        await self._event_bus.dispatch(AgentInterrupted())

    async def on_subagent_called(self, agent_name: str, task: str) -> None:
        await self._event_bus.dispatch(SubagentCalled(agent_name=agent_name, task=task))

    async def on_agent_error(
        self, type: str, message: str, code: str | None, param: str | None
    ) -> None:
        await self._event_bus.dispatch(
            AgentError(type=type, message=message, code=code, param=param)
        )