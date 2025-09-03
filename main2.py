"""
Simple Voice Assistant using OpenAI Realtime API directly.
No complex state machine - just direct API usage.
"""

import asyncio
from typing import Optional

from agent.realtime.realtime_api import OpenAIRealtimeAPI
from agent.realtime.views import VoiceAssistantConfig
from agent.realtime.websocket_manager import WebSocketManager
from audio.capture import AudioCapture
from audio.sound_player import SoundPlayer
from shared.logging_mixin import LoggingMixin


class SimpleVoiceAssistant(LoggingMixin):
    """Simple voice assistant that directly uses OpenAI Realtime API."""

    def __init__(self, config: Optional[VoiceAssistantConfig] = None):
        self.config = config or VoiceAssistantConfig()

        # Basic services
        self.sound_player = SoundPlayer()
        self.audio_capture = AudioCapture()

        # Realtime API - direct usage
        self.realtime_api = OpenAIRealtimeAPI(
            realtime_config=self.config,
            ws_manager=WebSocketManager.for_gpt_realtime(),
            audio_capture=self.audio_capture,
        )

        self._running = False
        self.logger.info("Simple Voice Assistant initialized")

    async def start(self) -> None:
        """Start the voice assistant"""
        if self._running:
            self.logger.warning("Assistant already running")
            return

        self._running = True
        self.logger.info("Starting Simple Voice Assistant")
        self.sound_player.play_startup_sound()

        try:
            # Start audio capture
            self.audio_capture.start_stream()
            self.logger.info("Audio capture started")

            # Run realtime session directly
            self.logger.info("Starting realtime session...")
            success = await self.realtime_api.setup_and_run()

            if success:
                self.logger.info("Realtime session completed successfully")
            else:
                self.logger.error("Realtime session failed")

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception:
            self.logger.exception("Error in voice assistant")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the voice assistant"""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping Simple Voice Assistant")

        # Stop audio capture
        try:
            self.audio_capture.stop_stream()
            self.logger.info("Audio capture stopped")
        except Exception:
            self.logger.exception("Audio capture cleanup failed")

        # Stop sound playback
        try:
            self.sound_player.stop_sounds()
            self.logger.info("Sound playback stopped")
        except Exception:
            self.logger.exception("Sound cleanup failed")

        self.logger.info("Simple Voice Assistant stopped")


async def main():
    assistant = SimpleVoiceAssistant()
    try:
        await assistant.start()
    except KeyboardInterrupt:
        pass
    finally:
        await assistant.stop()


if __name__ == "__main__":
    asyncio.run(main())
