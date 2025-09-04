"""
Voice Assistant Controller with modular service architecture.
Clean separation of concerns with dependency injection and service factories.
"""

import asyncio
from typing import Optional

from agent.config.env import validate_environment_variables
from agent.config.loader import load_config_from_yaml
from agent.config.views import VoiceAssistantConfig
from agent.realtime.event_bus import EventBus
from agent.realtime.realtime_api import OpenAIRealtimeAPI
from agent.realtime.transcription.service import TranscriptionService
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.state.context import VoiceAssistantContext
from agent.state.timeout_service import TimeoutService
from audio.capture import AudioCapture
from audio.detection import AudioDetectionService
from audio.sound_player import SoundPlayer
from audio.wake_word_listener import WakeWordListener
from shared.logging_mixin import LoggingMixin


class ServiceFactory:
    """Factory für Service-Erstellung basierend auf Config."""

    @staticmethod
    def create_wake_word_listener(config: VoiceAssistantConfig) -> WakeWordListener:
        return WakeWordListener(
            wakeword=config.wake_word.keyword,
            sensitivity=config.wake_word.sensitivity,
        )

    @staticmethod
    def create_realtime_api(
        config: VoiceAssistantConfig, audio_capture: AudioCapture
    ) -> OpenAIRealtimeAPI:
        return OpenAIRealtimeAPI(
            voice_assistant_config=config,
            ws_manager=WebSocketManager.from_model(model=config.agent.model),
            audio_capture=audio_capture,
            transcription_service=TranscriptionService(),
        )


class ServiceContainer:
    """Container für alle Services mit sauberer Initialisierung."""

    def __init__(self, config: VoiceAssistantConfig):
        self.sound_player = SoundPlayer()
        self.audio_capture = AudioCapture()
        self.event_bus = EventBus()

        # Config-basierte Services
        self.wake_word_listener = ServiceFactory.create_wake_word_listener(config)
        self.audio_detection_service = AudioDetectionService(
            audio_capture=self.audio_capture
        )
        self.timeout_service = TimeoutService(timeout_seconds=10.0)
        self.realtime_api = ServiceFactory.create_realtime_api(
            config, self.audio_capture
        )

        # Context
        self.context = VoiceAssistantContext(
            wake_word_listener=self.wake_word_listener,
            audio_capture=self.audio_capture,
            audio_detection_service=self.audio_detection_service,
            timeout_service=self.timeout_service,
            event_bus=self.event_bus,
            realtime_api=self.realtime_api,
        )


class VoiceAssistantController(LoggingMixin):
    """Main controller coordinating states and services for voice assistant functionality."""

    def __init__(self, config: VoiceAssistantConfig):
        self._config = config
        self._services = ServiceContainer(config)
        self._running = False
        self._shutdown_event = asyncio.Event()

        self.logger.info("Voice Assistant Controller initialized")

    @property
    def context(self) -> VoiceAssistantContext:
        """Zugriff auf den Voice Assistant Context."""
        return self._services.context

    async def start(self) -> None:
        """Start the voice assistant."""
        if self._running:
            self.logger.warning("Controller already running")
            return

        self.logger.info("Starting Voice Assistant Controller")
        self._running = True
        self._services.sound_player.play_startup_sound()

        try:
            await self._run_main_loop()
        except Exception as e:
            self.logger.exception(f"Controller error: {e}")
        finally:
            await self._cleanup()

    async def _run_main_loop(self) -> None:
        """Hauptschleife des Controllers."""
        # Initialize initial state
        await self.context.state.on_enter(self.context)

        # Event-driven loop statt polling
        while self._running:
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.1)
                break  # Shutdown requested
            except asyncio.TimeoutError:
                continue  # Normal operation
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested by user")
                break

    async def stop(self) -> None:
        """Stop the voice assistant gracefully."""
        if not self._running:
            return

        self.logger.info("Stopping Voice Assistant Controller")
        self._running = False
        self._shutdown_event.set()

    async def _cleanup(self) -> None:
        """Cleanup all services and resources."""
        self.logger.info("Cleaning up services...")

        cleanup_tasks = [
            self._cleanup_state(),
            self._cleanup_wake_word_listener(),
            self._cleanup_sound_player(),
        ]

        # Parallel cleanup mit Fehlerbehandlung
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_name = ["state", "wake_word", "sound_player"][i]
                self.logger.exception(f"Error cleaning up {service_name}: {result}")

        self.logger.info("Voice Assistant Controller stopped")

    async def _cleanup_state(self) -> None:
        """Cleanup current state."""
        try:
            await self.context.state.on_exit(self.context)
        except Exception as e:
            raise RuntimeError(f"State cleanup failed: {e}")

    async def _cleanup_wake_word_listener(self) -> None:
        """Cleanup wake word listener."""
        try:
            self._services.wake_word_listener.cleanup()
        except Exception as e:
            raise RuntimeError(f"WakeWord cleanup failed: {e}")

    async def _cleanup_sound_player(self) -> None:
        """Cleanup sound player."""
        try:
            self._services.sound_player.stop_sounds()
        except Exception as e:
            raise RuntimeError(f"Sound cleanup failed: {e}")


class VoiceAssistantApp:
    """Application entry point with proper lifecycle management."""

    def __init__(self):
        self._controller: Optional[VoiceAssistantController] = None

    async def run(self) -> None:
        """Run the voice assistant application."""
        try:
            # Setup
            validate_environment_variables()
            config = load_config_from_yaml()

            # Run
            self._controller = VoiceAssistantController(config)
            await self._controller.start()

        except KeyboardInterrupt:
            pass  # Graceful shutdown
        except Exception:
            print("Critical application error")
            raise
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """Graceful application shutdown."""
        if self._controller:
            await self._controller.stop()


async def main():
    """Application entry point."""
    app = VoiceAssistantApp()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
