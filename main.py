"""
Voice Assistant Controller with modular service architecture.
States and services manage their own responsibilities for clean separation of concerns.
"""

import asyncio
from typing import Optional

from agent.realtime.event_bus import EventBus
from agent.realtime.realtime_api import OpenAIRealtimeAPI
from agent.realtime.views import VoiceAssistantConfig
from agent.realtime.websocket_manager import WebSocketManager
from agent.state.context import VoiceAssistantContext
from audio.capture import AudioCapture
from audio.detection import AudioDetectionService
from audio.sound_player import SoundPlayer
from audio.wake_word_listener import WakeWordListener
from agent.state.timeout_service import TimeoutService
from shared.logging_mixin import LoggingMixin


class VoiceAssistantController(LoggingMixin):
    """Main controller coordinating states and services for voice assistant functionality."""

    def __init__(self, config: Optional[VoiceAssistantConfig] = None):
        self.config = config or VoiceAssistantConfig()

        # Services
        self.sound_player = SoundPlayer()
        self.audio_capture = AudioCapture()
        self.event_bus = EventBus()

        self.wake_word_listener = WakeWordListener(
            wakeword=self.config.wake_word,
            sensitivity=self.config.wakeword_sensitivity,
        )
        self.audio_detection_service = AudioDetectionService(
            audio_capture=self.audio_capture
        )
        self.timeout_service = TimeoutService(timeout_seconds=10.0)

        self.realtime_api = OpenAIRealtimeAPI(
            realtime_config=VoiceAssistantConfig(),
            ws_manager=WebSocketManager.for_gpt_realtime(),
            audio_capture=self.audio_capture,
        )

        # Context with dependencies
        self.context = VoiceAssistantContext(
            wake_word_listener=self.wake_word_listener,
            audio_capture=self.audio_capture,
            audio_detection_service=self.audio_detection_service,
            timeout_service=self.timeout_service,
            realtime_api=self.realtime_api,
            event_bus=self.event_bus,
        )

        self._running = False

        self.logger.info("Voice Assistant Controller initialized (slim mode)")

    async def start(self) -> None:
        """Start the voice assistant"""
        if self._running:
            self.logger.warning("Controller already running")
            return

        self._running = True
        self.logger.info("Starting Voice Assistant Controller")
        self.sound_player.play_startup_sound()

        try:
            # Initialize the initial state (IdleState will start wake word detection)
            await self.context.state.on_enter(self.context)

            # Simple idle loop - states manage themselves
            while self._running:
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception:
            self.logger.exception("Unhandled error in controller")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the voice assistant"""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping Voice Assistant Controller")

        try:
            # Clean up current state (will stop any running tasks)
            await self.context.state.on_exit(self.context)
        except Exception:
            self.logger.exception("Error during state cleanup")

        # Clean up services
        try:
            self.wake_word_listener.cleanup()
        except Exception:
            self.logger.exception("WakeWord cleanup failed")

        try:
            self.sound_player.stop_sounds()
        except Exception:
            self.logger.exception("Sound stop failed")

        self.logger.info("Voice Assistant Controller stopped")


async def main():
    ctrl = VoiceAssistantController()
    try:
        await ctrl.start()
    except KeyboardInterrupt:
        pass
    finally:
        await ctrl.stop()


if __name__ == "__main__":
    asyncio.run(main())
