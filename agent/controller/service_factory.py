import asyncio
from dataclasses import dataclass
from agent.config.views import AgentConfig, WakeWordConfig, VoiceAssistantConfig
from agent.realtime.event_bus import EventBus
from agent.realtime.reatlime_client import RealtimeClient
from agent.realtime.tools.views import SpecialToolParameters
from agent.realtime.tools.tools import Tools
from agent.state.context import VoiceAssistantContext
from audio.capture import AudioCapture
from audio.detection import AudioDetectionService
from audio.player.audio_manager import AudioManager
from audio.sound_event_handler import SoundEventHandler
from audio.wake_word_listener import WakeWordListener


@dataclass
class ServiceBundle:
    """All created services bundled together"""

    event_bus: EventBus
    audio_capture: AudioCapture
    audio_manager: AudioManager
    audio_detection_service: AudioDetectionService
    wake_word_listener: WakeWordListener
    sound_event_handler: SoundEventHandler
    realtime_client: RealtimeClient
    context: VoiceAssistantContext


class ServiceFactory:
    """Factory for creating all voice assistant services."""

    def __init__(
        self, agent_config: AgentConfig, wake_word_config: WakeWordConfig, tools: Tools
    ):
        self.agent_config = agent_config
        self.wake_word_config = wake_word_config
        self.tools = tools
        self.event_bus = EventBus()  # Created once, shared everywhere
        self.event_bus.attach_loop(asyncio.get_running_loop())

    def create_services(self) -> ServiceBundle:
        """Create all services with proper dependencies."""
        # Core audio services
        audio_capture = self._create_audio_capture()
        audio_manager = self._create_audio_manager()

        # Audio detection and wake word
        audio_detection_service = self._create_audio_detection_service(audio_capture)
        wake_word_listener = self._create_wake_word_listener()

        # Sound handling
        sound_event_handler = self._create_sound_event_handler(audio_manager)

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
        return AudioManager(event_bus=self.event_bus)

    def _create_audio_detection_service(
        self, audio_capture: AudioCapture
    ) -> AudioDetectionService:
        return AudioDetectionService(
            audio_capture=audio_capture,
            event_bus=self.event_bus,
        )

    def _create_wake_word_listener(self) -> WakeWordListener:
        return WakeWordListener(
            wakeword=self.wake_word_config.keyword,
            sensitivity=self.wake_word_config.sensitivity,
            event_bus=self.event_bus,
        )

    def _create_sound_event_handler(
        self, audio_manager: AudioManager
    ) -> SoundEventHandler:
        return SoundEventHandler(audio_manager.strategy, self.event_bus)

    # Hier müssen die neuen Tools übergeben werden denke ich :)
    def _create_realtime_client(
        self, audio_capture: AudioCapture, audio_manager: AudioManager
    ) -> RealtimeClient:
        # Create special tool parameters bundle
        special_tool_parameters = SpecialToolParameters(
            audio_manager=audio_manager,
            event_bus=self.event_bus,
            agent_config=self.agent_config,
        )

        return RealtimeClient(
            voice_assistant_config=VoiceAssistantConfig(
                agent=self.agent_config,
                wake_word=self.wake_word_config,
            ),
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
