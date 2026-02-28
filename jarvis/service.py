import asyncio
import logging

from jarvis.events import EventBus, AgentEventAdapter
from jarvis.events.views import WakeWordDetected
from jarvis.wake_word import WakeWord, WakeWordListener
from jarvis.watchdogs import SoundEffectWatchdog, LightsWatchdog

from rtvoice import (
    RealtimeAgent,
    RealtimeModel,
    SubAgent,
    AssistantVoice,
    Tools,
)
from rtvoice.views import NoiseReduction
from rtvoice.audio import AudioOutputDevice
from rtvoice.mcp import MCPServer

logger = logging.getLogger(__name__)


class Jarvis:
    def __init__(
        self,
        realtime_model: RealtimeModel = RealtimeModel.GPT_REALTIME_MINI,
        voice: AssistantVoice = AssistantVoice.MARIN,
        instructions: str = "",
        tools: Tools | None = None,
        subagents: list[SubAgent] | None = None,
        mcp_servers: list[MCPServer] | None = None,
        wake_word: WakeWord = WakeWord.HEY_JARVIS,
        wake_word_sensitivity: float = 0.8,
        noise_reduction: NoiseReduction = NoiseReduction.FAR_FIELD,
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
        self._agent_listener = AgentEventAdapter(event_bus=self._event_bus)
        self._agent: RealtimeAgent | None = None
        self._next_agent: RealtimeAgent | None = None
        self._prepared: bool = False

        self._wake_word_listener = WakeWordListener(
            wake_word=wake_word,
            sensitivity=wake_word_sensitivity,
            on_detection=self._on_wake_word_detected,
        )
        self._sound_effect_watchdog = SoundEffectWatchdog(event_bus=self._event_bus)
        self._lights_watchdog = LightsWatchdog(event_bus=self._event_bus)

    def _create_agent(self) -> RealtimeAgent:
        return RealtimeAgent(
            instructions=self._instructions,
            model=self._realtime_model,
            voice=self._voice,
            tools=self._tools,
            subagents=self._subagents,
            mcp_servers=self._mcp_servers,
            noise_reduction=self._noise_reduction,
            audio_output=self._audio_output_device,
            agent_listener=self._agent_listener,
        )

    async def _prepare_next_agent(self) -> None:
        agent = self._create_agent()
        await agent.prepare()
        self._next_agent = agent
        logger.debug("Next agent pre-prepared and ready")

    async def _on_wake_word_detected(self) -> None:
        logger.info("Wake word detected – dispatching event...")
        await self._event_bus.dispatch(WakeWordDetected())

        self._agent = self._next_agent or self._create_agent()
        self._next_agent = None

        await self._agent.run()

        self._agent = None
        asyncio.create_task(self._prepare_next_agent())

    async def prepare(self) -> None:
        """Pre-warms the agent and starts background services.

        Optional but recommended: call this before run() to reduce latency
        on the first wake-word detection. Safe to call multiple times –
        subsequent calls are no-ops.
        """
        if self._prepared:
            return

        logger.info("Preparing Jarvis...")
        await self._lights_watchdog.start()
        await self._prepare_next_agent()
        self._prepared = True
        logger.info("Jarvis prepared and ready")

    async def run(self) -> None:
        """Starts the wake-word listener loop.

        Calls prepare() automatically if not already prepared, so calling
        prepare() beforehand is optional but gives faster first-response time.
        """
        await self.prepare()

        while True:
            await self._wake_word_listener.listen()

    async def stop(self) -> None:
        if self._is_running():
            await self._agent.stop()

    def _is_running(self) -> bool:
        return self._agent is not None