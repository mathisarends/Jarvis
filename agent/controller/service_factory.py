import asyncio
from dataclasses import dataclass
from agent.config.views import (
    AgentConfig,
    AssistantAudioConfig,
    WakeWordConfig,
)
from agent.realtime.event_bus import EventBus
from agent.realtime.events.client.session_update import InputAudioTranscriptionConfig
from agent.realtime.reatlime_client import RealtimeClient
from agent.realtime.tools.tools import Tools
from agent.realtime.tools.views import SpecialToolParameters
from agent.state.context import VoiceAssistantContext
from audio.capture import AudioCapture
from audio.detection import AudioDetectionService
from audio.player.audio_manager import AudioManager
from audio.sound_event_handler import SoundEventHandler
from audio.wake_word_listener import WakeWordListener


@dataclass
class ServiceBundle:
    """All created services bundled together

    Note: `wake_word_listener` and `audio_detection_service` may be None when
    the corresponding features are not configured. Callers should treat them
    as optional.
    """

    event_bus: EventBus
    audio_capture: AudioCapture
    audio_manager: AudioManager
    sound_event_handler: SoundEventHandler
    realtime_client: RealtimeClient
    context: VoiceAssistantContext

    audio_detection_service: AudioDetectionService | None
    wake_word_listener: WakeWordListener | None


class ServiceFactory:
    """Factory for creating all voice assistant services."""

    def __init__(
        self,
        agent_config: AgentConfig,
        assistant_audio_config: AssistantAudioConfig,
        tools: Tools,
        wake_word_config: WakeWordConfig | None = None,
        transcription_config: InputAudioTranscriptionConfig | None = None,
    ):
        self.agent_config = agent_config
        self.assistant_audio_config = assistant_audio_config
        self.tools = tools

        self.wake_word_config = wake_word_config
        self.transcription_config = transcription_config
        self.event_bus = EventBus()  # Created once, shared everywhere
        self.event_bus.attach_loop(asyncio.get_running_loop())

    def create_services(self) -> ServiceBundle:
        """Create all services with proper dependencies."""
        # Core audio services
        audio_capture = self._create_audio_capture()
        audio_manager = self._create_audio_manager()

        # Audio detection and wake word
        audio_detection_service = self._create_audio_detection_service(audio_capture)
        wake_word_listener = self._create_wake_word_listener_if_configured()

        # Sound handling
        sound_event_handler = self._create_sound_event_handler(audio_manager)

        # Set EventBus here for simplified interface
        self.assistant_audio_config.audio_playback_strategy.set_event_bus(
            self.event_bus
        )

        # Realtime AI client
        realtime_client = self._create_realtime_client(audio_capture, audio_manager)

        # Context (state machine)
        context = self._create_context(
            wake_word_listener,
            audio_capture,
            audio_detection_service,
            audio_manager,
            realtime_client,
        )

        return ServiceBundle(
            event_bus=self.event_bus,
            audio_capture=audio_capture,
            audio_manager=audio_manager,
            audio_detection_service=audio_detection_service,
            wake_word_listener=wake_word_listener,
            sound_event_handler=sound_event_handler,
            realtime_client=realtime_client,
            context=context,
        )

    def _create_audio_capture(self) -> AudioCapture:
        return AudioCapture()

    def _create_audio_manager(self) -> AudioManager:
        return AudioManager(self.assistant_audio_config.audio_playback_strategy)

    def _create_audio_detection_service(
        self, audio_capture: AudioCapture
    ) -> AudioDetectionService:
        return AudioDetectionService(
            audio_capture=audio_capture,
            event_bus=self.event_bus,
        )

    def _create_wake_word_listener_if_configured(self) -> WakeWordListener:
        # Create wake-word listener only if a WakeWordConfig was provided
        if not self.wake_word_config:
            return None

        return WakeWordListener(
            wakeword=self.wake_word_config.keyword,
            sensitivity=self.wake_word_config.sensitivity,
            event_bus=self.event_bus,
        )

    def _create_sound_event_handler(
        self, audio_manager: AudioManager
    ) -> SoundEventHandler:
        return SoundEventHandler(audio_manager.strategy, self.event_bus)

    def _create_realtime_client(
        self, audio_capture: AudioCapture, audio_manager: AudioManager
    ) -> RealtimeClient:
        special_tool_parameters = SpecialToolParameters(
            audio_manager=audio_manager,
            event_bus=self.event_bus,
            assistant_audio_config=self.assistant_audio_config,
            tool_calling_model_name=self.agent_config.tool_calling_model_name,
        )

        return RealtimeClient(
            agent_config=self.agent_config,
            assistant_audio_config=self.assistant_audio_config,
            audio_capture=audio_capture,
            special_tool_parameters=special_tool_parameters,
            event_bus=self.event_bus,
            tools=self.tools,
        )

    def _create_context(
        self,
        wake_word_listener: WakeWordListener,
        audio_capture: AudioCapture,
        audio_detection_service: AudioDetectionService,
        audio_manager: AudioManager,
        realtime_client: RealtimeClient,
    ) -> VoiceAssistantContext:
        return VoiceAssistantContext(
            wake_word_listener=wake_word_listener,
            audio_capture=audio_capture,
            audio_detection_service=audio_detection_service,
            audio_manager=audio_manager,
            event_bus=self.event_bus,
            realtime_client=realtime_client,
        )
