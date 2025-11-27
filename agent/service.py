import asyncio

from agent.config import AgentEnv
from agent.config.models import (
    ModelSettings,
    VoiceSettings,
    TranscriptionSettings,
    WakeWordSettings,
)
from agent.realtime.event_bus import EventBus
from agent.realtime.reatlime_client import RealtimeClient
from agent.realtime.tools.tools import Tools
from agent.realtime.tools.views import SpecialToolParameters
from agent.state.context import VoiceAssistantContext
from agent.wake_word import WakeWordListener
from agent.mic import MicrophoneCapture, SpeechDetector
from agent.sound import AudioPlayer
from agent.sound.handler import SoundEventHandler
from shared.logging_mixin import LoggingMixin


class RealtimeAgent(LoggingMixin):
    def __init__(
        self,
        model_settings: ModelSettings | None = None,
        voice_settings: VoiceSettings | None = None,
        transcription_settings: TranscriptionSettings | None = None,
        wake_word_settings: WakeWordSettings | None = None,
        env: AgentEnv | None = None,
    ):
        self._env = env or AgentEnv()
        self._model_settings = model_settings or ModelSettings()
        self._voice_settings = voice_settings or VoiceSettings()
        self._transcription_settings = transcription_settings or TranscriptionSettings()
        self._wake_word_settings = wake_word_settings or WakeWordSettings()

        self._tools = Tools(mcp_tools=self._model_settings.mcp_tools)

        self._event_bus = self._create_event_bus()
        self._audio_capture = self._create_audio_capture()
        self._audio_player = self._create_audio_player()
        self._speech_detector = self._create_speech_detector()
        self._wake_word_listener = self._create_wake_word_listener()
        self._sound_event_handler = self._create_sound_event_handler()
        self._realtime_client = self._create_realtime_client()
        self._context = self._create_context()

        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        if self._running:
            self.logger.warning("Agent already running")
            return

        try:
            self.logger.info("Starting Voice Assistant")
            self._running = True

            await self._context.run()

            while self._running:
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.1)
                    break
                except asyncio.TimeoutError:
                    continue
                except KeyboardInterrupt:
                    self.logger.info("Shutdown requested by user")
                    break

        except KeyboardInterrupt:
            pass
        finally:
            await self._cleanup_all_services()

    async def stop(self) -> None:
        if not self._running:
            return

        self.logger.info("Stopping Voice Assistant")
        self._running = False
        self._shutdown_event.set()

    async def _cleanup_all_services(self) -> None:
        self.logger.info("Cleaning up all services...")

        cleanup_tasks = {
            "state_machine": self._cleanup_state_machine(),
            "wake_word": self._cleanup_wake_word_service(),
            "sound": self._cleanup_sound_service(),
        }

        results = await asyncio.gather(
            *cleanup_tasks.values(), 
            return_exceptions=True
        )

        for service_name, result in zip(cleanup_tasks.keys(), results):
            if isinstance(result, Exception):
                self.logger.exception("Error cleaning up %s", service_name, exc_info=result)

        self.logger.info("All services cleaned up")

    async def _cleanup_state_machine(self) -> None:
        await self._context.state.on_exit(self._context)

    async def _cleanup_wake_word_service(self) -> None:
        if self._wake_word_listener:
            self._wake_word_listener.cleanup()

    async def _cleanup_sound_service(self) -> None:
        self._audio_player.stop_sounds()

    def _create_event_bus(self) -> EventBus:
        event_bus = EventBus()
        event_bus.attach_loop(asyncio.get_running_loop())
        return event_bus

    def _create_audio_capture(self) -> MicrophoneCapture:
        return MicrophoneCapture()

    def _create_audio_player(self) -> AudioPlayer:
        return AudioPlayer(self._voice_settings.playback_strategy)

    def _create_speech_detector(self) -> SpeechDetector:
        return SpeechDetector(
            audio_capture=self._audio_capture,
            event_bus=self._event_bus,
        )

    def _create_wake_word_listener(self) -> WakeWordListener | None:
        if not self._wake_word_settings.enabled:
            return None

        return WakeWordListener(
            wakeword=self._wake_word_settings.keyword,
            sensitivity=self._wake_word_settings.sensitivity,
            event_bus=self._event_bus,
        )

    def _create_sound_event_handler(self) -> SoundEventHandler:
        return SoundEventHandler(self._audio_player, self._event_bus)

    def _create_realtime_client(self) -> RealtimeClient:
        self._voice_settings.playback_strategy.set_event_bus(self._event_bus)

        special_tool_parameters = SpecialToolParameters(
            audio_player=self._audio_player,
            event_bus=self._event_bus,
            voice_settings=self._voice_settings,
            tool_calling_model_name=self._model_settings.tool_calling_model_name,
        )

        return RealtimeClient(
            model_settings=self._model_settings,
            voice_settings=self._voice_settings,
            audio_capture=self._audio_capture,
            special_tool_parameters=special_tool_parameters,
            event_bus=self._event_bus,
            tools=self._tools,
        )

    def _create_context(self) -> VoiceAssistantContext:
        return VoiceAssistantContext(
            wake_word_listener=self._wake_word_listener,
            audio_capture=self._audio_capture,
            speech_detector=self._speech_detector,
            audio_player=self._audio_player,
            event_bus=self._event_bus,
            realtime_client=self._realtime_client,
        )