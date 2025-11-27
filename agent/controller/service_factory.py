import asyncio
from dataclasses import dataclass

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
from agent.mic import MicrophoneCapture
from audio.detection import AudioDetectionService
from audio.player.audio_manager import AudioManager
from audio.sound_event_handler import SoundEventHandler


@dataclass
class ServiceBundle:
    event_bus: EventBus
    audio_capture: MicrophoneCapture
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
        model_settings: ModelSettings,
        voice_settings: VoiceSettings,
        transcription_settings: TranscriptionSettings,
        wake_word_settings: WakeWordSettings,
        tools: Tools,
    ):
        self.model_settings = model_settings
        self.voice_settings = voice_settings
        self.transcription_settings = transcription_settings
        self.wake_word_settings = wake_word_settings
        self.tools = tools

        self.event_bus = EventBus()
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
        self.voice_settings.playback_strategy.set_event_bus(
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

    def _create_audio_capture(self) -> MicrophoneCapture:
        return MicrophoneCapture()

    def _create_audio_manager(self) -> AudioManager:
        return AudioManager(self.voice_settings.playback_strategy)

    def _create_audio_detection_service(
        self, audio_capture: MicrophoneCapture
    ) -> AudioDetectionService:
        return AudioDetectionService(
            audio_capture=audio_capture,
            event_bus=self.event_bus,
        )

    def _create_wake_word_listener_if_configured(self) -> WakeWordListener:
        if not self.wake_word_settings.enabled:
            return None

        return WakeWordListener(
            wakeword=self.wake_word_settings.keyword,
            sensitivity=self.wake_word_settings.sensitivity,
            event_bus=self.event_bus,
        )

    def _create_sound_event_handler(
        self, audio_manager: AudioManager
    ) -> SoundEventHandler:
        return SoundEventHandler(audio_manager.strategy, self.event_bus)

    def _create_realtime_client(
        self, audio_capture: MicrophoneCapture, audio_manager: AudioManager
    ) -> RealtimeClient:
        special_tool_parameters = SpecialToolParameters(
            audio_manager=audio_manager,
            event_bus=self.event_bus,
            voice_settings=self.voice_settings,
            tool_calling_model_name=self.model_settings.tool_calling_model_name,
        )

        return RealtimeClient(
            model_settings=self.model_settings,
            voice_settings=self.voice_settings,
            audio_capture=audio_capture,
            special_tool_parameters=special_tool_parameters,
            event_bus=self.event_bus,
            tools=self.tools,
        )

    def _create_context(
        self,
        wake_word_listener: WakeWordListener,
        audio_capture: MicrophoneCapture,
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
