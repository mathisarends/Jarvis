import logging

from jarvis.events import EventBus
from jarvis.events.views import (
    WakeWordDetected,
    AgentStarted,
    AgentStopped,
    AgentInterrupted,
)
from jarvis.wake_word import WakeWord, WakeWordListener
from jarvis.watchdogs import SoundEffectWatchdog, LightsWatchdog

from rtvoice import (
    RealtimeAgent, 
    RealtimeModel, 
    AgentListener, 
    SubAgent,
    AssistantVoice,
    Tools,
)
from rtvoice.views import NoiseReduction
from rtvoice.audio import AudioOutputDevice

from rtvoice.mcp import MCPServer

logger = logging.getLogger(__name__)

class Jarvis(AgentListener):
    def __init__(
        self,
        realtime_model: RealtimeModel = RealtimeModel.GPT_REALTIME_MINI,
        voice: AssistantVoice = AssistantVoice.MARIN,
        instructions: str = "",
        tools: Tools | None = None,
        subagents: list[SubAgent] | None = None,
        mcp_servers: list[MCPServer] | None = None,
        wake_word: WakeWord = WakeWord.JARVIS,
        wake_word_sensitivity: float = 0.8,
        noise_reduction: NoiseReduction = NoiseReduction.FAR_FIELD,
        access_key: str | None = None,
        audio_output_device: AudioOutputDevice | None = None,
    ) -> None:
        self._realtime_model = realtime_model
        self._voice = voice
        self._instructions = instructions
        self._tools = tools
        self._subagents = subagents or []
        self._mcp_servers = mcp_servers or []
        self._noise_reduction = noise_reduction
        self._audio_output_device = audio_output_device

        self._event_bus = EventBus()
        self._agent: RealtimeAgent | None = None

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
        self._agent = RealtimeAgent(
            instructions=self._instructions,
            model=self._realtime_model,
            voice=self._voice,
            tools=self._tools,
            subagents=self._subagents,
            mcp_servers=self._mcp_servers,
            noise_reduction=self._noise_reduction,
            audio_output=self._audio_output_device,
            agent_listener=self,
        )
        async with self._agent:
            pass
        self._agent = None

    async def on_agent_started(self) -> None:
        logger.info("Agent started – dispatching event...")
        await self._event_bus.dispatch(AgentStarted())

    async def on_agent_stopped(self) -> None:
        logger.info("Agent stopped – dispatching event...")
        await self._event_bus.dispatch(AgentStopped())

    async def on_agent_interrupted(self) -> None:
        logger.info("Agent interrupted – dispatching event...")
        await self._event_bus.dispatch(AgentInterrupted())

    async def run(self) -> None:
        while True:
            await self._listener.listen()

    async def stop(self) -> None:
        if self._is_running():
            await self._agent.stop()

    def _is_running(self) -> bool:
        return self._agent is not None
