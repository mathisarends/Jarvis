"""
Voice Assistant Controller - All-in-one clean implementation.
"""

import asyncio

from agent.config.views import VoiceAssistantConfig
from agent.realtime.event_bus import EventBus
from agent.realtime.realtime_api import OpenAIRealtimeAPI
from agent.realtime.transcription.service import TranscriptionService
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.state.context import VoiceAssistantContext
from agent.state.timeout_service import TimeoutService
from audio.capture import AudioCapture
from audio.detection import AudioDetectionService
from audio.player.audio_manager import AudioManager
from audio.sound_event_handler import SoundEventHandler
from audio.wake_word_listener import WakeWordListener
from shared.logging_mixin import LoggingMixin


class VoiceAssistantController(LoggingMixin):
    """
    Voice Assistant Controller - manages all services and application lifecycle.
    Single responsibility: coordinate voice assistant functionality.
    """

    def __init__(self, config: VoiceAssistantConfig):
        self._config = config
        self._running = False
        self._shutdown_event = asyncio.Event()

        self._init_core_services()
        self._init_audio_services()
        self._init_ai_services()
        self._init_context()

        self.logger.info("Voice Assistant Controller initialized")

    async def start(self) -> None:
        """Start the voice assistant."""
        if self._running:
            self.logger.warning("Controller already running")
            return

        self.logger.info("Starting Voice Assistant Controller")
        self._running = True
        self.audio_strategy.play_startup_sound()

        try:
            await self._run_application()
        except Exception as e:
            self.logger.exception(f"Controller error: {e}")
        finally:
            await self._cleanup_all_services()

    async def stop(self) -> None:
        """Stop the voice assistant gracefully."""
        if not self._running:
            return

        self.logger.info("Stopping Voice Assistant Controller")
        self._running = False
        self._shutdown_event.set()

    def _init_core_services(self) -> None:
        """Initialize core services."""
        self.event_bus = EventBus()

        self.audio_manager = AudioManager()
        self.audio_strategy = self.audio_manager.strategy
        self.sound_event_handler = SoundEventHandler(
            self.audio_strategy, self.event_bus
        )
        self.timeout_service = TimeoutService(timeout_seconds=10.0)

    def _init_audio_services(self) -> None:
        """Initialize audio-related services."""
        self.audio_capture = AudioCapture()
        self.audio_detection_service = AudioDetectionService(
            audio_capture=self.audio_capture
        )
        self.wake_word_listener = WakeWordListener(
            wakeword=self._config.wake_word.keyword,
            sensitivity=self._config.wake_word.sensitivity,
        )

    def _init_ai_services(self) -> None:
        """Initialize AI and realtime services."""
        self.realtime_api = OpenAIRealtimeAPI(
            voice_assistant_config=self._config,
            ws_manager=WebSocketManager.from_model(model=self._config.agent.model),
            audio_capture=self.audio_capture,
            transcription_service=TranscriptionService(),
            audio_manager=self.audio_manager,
            event_bus=self.event_bus,
        )

    def _init_context(self) -> None:
        """Initialize the voice assistant context."""
        self.context = VoiceAssistantContext(
            wake_word_listener=self.wake_word_listener,
            audio_capture=self.audio_capture,
            audio_detection_service=self.audio_detection_service,
            timeout_service=self.timeout_service,
            audio_manager=self.audio_manager,
            event_bus=self.event_bus,
            realtime_api=self.realtime_api,
        )

    async def _run_application(self) -> None:
        """Main application loop."""
        # Start the state machine
        await self.context.state.on_enter(self.context)

        # Event-driven loop
        while self._running:
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.1)
                break  # Shutdown requested
            except asyncio.TimeoutError:
                continue  # Normal operation
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested by user")
                break

    async def _cleanup_all_services(self) -> None:
        """Cleanup all services in parallel."""
        self.logger.info("Cleaning up all services...")

        cleanup_tasks = [
            self._cleanup_state_machine(),
            self._cleanup_wake_word_service(),
            self._cleanup_sound_service(),
        ]

        # Run cleanup tasks in parallel
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Log any cleanup errors
        service_names = ["state_machine", "wake_word", "sound"]
        for service_name, result in zip(service_names, results):
            if isinstance(result, Exception):
                self.logger.exception(f"Error cleaning up {service_name}: {result}")

        self.logger.info("All services cleaned up")

    async def _cleanup_state_machine(self) -> None:
        """Cleanup the state machine."""
        await self.context.state.on_exit(self.context)

    async def _cleanup_wake_word_service(self) -> None:
        """Cleanup wake word detection."""
        self.wake_word_listener.cleanup()

    async def _cleanup_sound_service(self) -> None:
        """Cleanup sound playback."""
        self.audio_strategy.stop_sounds()
