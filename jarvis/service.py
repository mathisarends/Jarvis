import asyncio
from datetime import datetime
import logging

from jarvis.audio import VolumeSpeakerOutput
from jarvis.events import EventBus, AgentEventAdapter
from jarvis.events.views import WakeWordDetectedEvent, AgentStopCommand, AgentStoppedEvent
from jarvis.views import JarvisContext
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
    ) -> None:
        self._realtime_model = realtime_model
        self._voice = voice
        self._instructions = instructions
        self._tools = tools or Tools()
        self._subagents = subagents or []
        self._mcp_servers = mcp_servers or []
        self._noise_reduction = noise_reduction

        self._speaker = VolumeSpeakerOutput()
        self._event_bus = EventBus()
        self._context = JarvisContext(
            event_bus=self._event_bus,
            speaker=self._speaker,
        )
        self._agent_listener = AgentEventAdapter(event_bus=self._event_bus)
        self._agent: RealtimeAgent | None = None
        self._next_agent: RealtimeAgent | None = None
        self._prepared: bool = False
        self._background_tasks: set[asyncio.Task] = set()

        self._wake_word_listener = WakeWordListener(
            wake_word=wake_word,
            sensitivity=wake_word_sensitivity,
            on_detection=self._on_wake_word_detected,
        )
        self._sound_effect_watchdog = SoundEffectWatchdog(event_bus=self._event_bus)
        self._lights_watchdog = LightsWatchdog(event_bus=self._event_bus)

        self._event_bus.subscribe(AgentStopCommand, self._on_stop_command)
        self._register_core_tools()
        
    def _register_core_tools(self) -> None:
        @self._tools.action("Stop the current assistant run", suppress_response=True)
        async def stop_current_run(context: JarvisContext) -> None:
            await context.event_bus.dispatch(AgentStopCommand())

        @self._tools.action("Get the current local date and time.")
        def get_current_time() -> str:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        @self._tools.action("Set the speaker volume (0-100)")
        def set_volume(context: JarvisContext, percent: int) -> str:
            context.speaker.volume = percent
            return f"Volume set to {percent}%"

        @self._tools.action("Get the current speaker volume")
        def get_volume(context: JarvisContext) -> str:
            return f"Current volume: {context.speaker.volume}%"

    def _create_agent(self) -> RealtimeAgent:
        return RealtimeAgent(
            instructions=self._instructions,
            model=self._realtime_model,
            voice=self._voice,
            tools=self._tools,
            subagents=self._subagents,
            mcp_servers=self._mcp_servers,
            noise_reduction=self._noise_reduction,
            audio_output=self._speaker,
            agent_listener=self._agent_listener,
            context=self._context,
        )

    async def _prepare_next_agent(self) -> None:
        try:
            agent = self._create_agent()
            await agent.prepare()
            self._next_agent = agent
            logger.debug("Next agent pre-prepared and ready")
        except Exception:
            logger.exception("Failed to pre-prepare next agent – will create on demand")
            self._next_agent = None

    async def _on_wake_word_detected(self) -> None:
        logger.info("Wake word detected – dispatching event...")
        await self._event_bus.dispatch(WakeWordDetectedEvent())

        self._agent = self._next_agent or self._create_agent()
        self._next_agent = None

        try:
            await self._agent.run()
        except Exception:
            logger.exception("Agent session raised an unexpected error")
        finally:
            self._agent = None
            task = asyncio.create_task(self._prepare_next_agent())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

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
            try:
                await self._wake_word_listener.listen()
            except asyncio.CancelledError:
                logger.info("Jarvis run loop cancelled – shutting down")
                raise
            except Exception:
                logger.exception("Wake word listener error – restarting listener")

    async def _on_stop_command(self, _: AgentStopCommand) -> None:
        logger.info("Stop requested via command – stopping agent...")
        # Use ensure_future instead of await: stopping the agent cancels all running
        # tasks, including the tool-call task that triggered this handler. Awaiting
        # directly causes a recursive cancel loop. ensure_future defers the stop to
        # the next event loop tick, letting the tool call finish cleanly first.
        asyncio.ensure_future(self.stop())

    async def stop(self) -> None:
        if self._is_running():
            await self._agent.stop()
            await self._event_bus.dispatch(AgentStoppedEvent())

    def _is_running(self) -> bool:
        return self._agent is not None