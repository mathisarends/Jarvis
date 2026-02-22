from rtvoice import AgentListener

from rtvoice.events.bus import EventBus
from jarvis.events.views import AgentInterrupted, AgentStarted, AgentStopped


class AgentEventAdapter(AgentListener):
    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def on_agent_started(self) -> None:
        await self._event_bus.dispatch(AgentStarted())

    async def on_agent_stopped(self) -> None:
        await self._event_bus.dispatch(AgentStopped())

    async def on_agent_interrupted(self) -> None:
        await self._event_bus.dispatch(AgentInterrupted())