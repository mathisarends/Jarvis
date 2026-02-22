import logging

from jarvis.events import EventBus, AgentEventAdapter
from jarvis.events.views import (
     WakeWordDetected, 
     AgentStarted, 
     AgentStopped, 
     AgentInterrupted
)
from jarvis.wake_word import WakeWord, WakeWordListener
from jarvis.watchdogs import SoundEffectWatchdog, LightsWatchdog

from rtvoice import (
    RealtimeAgent, 
    RealtimeModel, 
    AgentListener, 
    SubAgent,
    AssistantVoice
)
from rtvoice.mcp import MCPServer

logger = logging.getLogger(__name__)


class Jarvis(AgentListener):
    def __init__(
        self,
        realtime_model: RealtimeModel = RealtimeModel.GPT_REALTIME,
        voice: AssistantVoice = AssistantVoice.MARIN,
        instructions: str = "",
        subagents: list[SubAgent] | None = None,
        mcp_servers: list[MCPServer] | None = None,
        wake_word: WakeWord = WakeWord.JARVIS,
        wake_word_sensitivity: float = 0.8,
        access_key: str | None = None,
    ) -> None:
        self._realtime_model = realtime_model
        self._voice = voice
        self._instructions = instructions
        self._subagents = subagents or []
        self._mcp_servers = mcp_servers or []

        self._event_bus = EventBus()
        self._agent_adapter = AgentEventAdapter(self._event_bus)

        self._listener = WakeWordListener(
            wake_word=wake_word,
            sensitivity=wake_word_sensitivity,
            access_key=access_key,
            on_detection=self._on_wake_word_detected,
        )
        self._sound_effect_watchdog = SoundEffectWatchdog(event_bus=self._event_bus)
        self._lights_watchdog = LightsWatchdog(event_bus=self._event_bus)

    async def _on_wake_word_detected(self) -> None:
        logger.info("Wake word detected – dispatching event...")
        await self._event_bus.dispatch(WakeWordDetected())

    async def on_agent_started(self) -> None:
        logger.info("Agent started – dispatching event...")
        await self._event_bus.dispatch(AgentStarted())

    async def on_agent_stopped(self) -> None:
        logger.info("Agent stopped – dispatching event...")
        await self._event_bus.dispatch(AgentStopped())

    async def on_agent_interrupted(self) -> None:
        logger.info("Agent interrupted – dispatching event...")
        await self._event_bus.dispatch(AgentInterrupted())

    def configure_agent(self) -> RealtimeAgent:
        return RealtimeAgent(
            instructions=self._instructions,
            model=self._realtime_model,
            subagents=self._subagents,
            mcp_servers=self._mcp_servers,
            agent_listener=self._agent_adapter
        )

    async def run(self) -> None:
        await self._listener.listen()

    # Implement agent running and stop for this facade here aswell (would be pretty nices)
    async def stop(self) -> None:
        ...