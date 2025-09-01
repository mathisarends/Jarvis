"""
Slimmed controller: States manage their own tasks
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from agent.realtime.views import AssistantVoice, RealtimeModel
from agent.state.base import VoiceAssistantContext, VoiceAssistantEvent
from audio import SoundFilePlayer
from audio.wake_word_listener import WakeWordListener, PorcupineBuiltinKeyword
from shared.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class VoiceAssistantConfig:
    voice: AssistantVoice = AssistantVoice.ALLOY
    realtime_model: RealtimeModel = RealtimeModel.GPT_REALTIME

    wake_word: PorcupineBuiltinKeyword = PorcupineBuiltinKeyword.PICOVOICE
    sensitivity: float = 0.7


class VoiceAssistantController(LoggingMixin):
    """Slim controller: States manage their own tasks."""

    def __init__(self, config: Optional[VoiceAssistantConfig] = None):
        self.config = config or VoiceAssistantConfig()

        # Services
        self.sound_player = SoundFilePlayer()
        self.wake_word_listener = WakeWordListener(
            wakeword=self.config.wake_word,
            sensitivity=self.config.sensitivity,
        )

        # Context with dependencies
        self.context = VoiceAssistantContext(
            sound_player=self.sound_player, wake_word_listener=self.wake_word_listener
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
            self.sound_player.stop_sound()
        except Exception:
            self.logger.exception("Sound stop failed")

        self.logger.info("Voice Assistant Controller stopped")

    async def handle_external_event(self, event: VoiceAssistantEvent) -> None:
        """
        Handle external events (if needed for testing or manual triggers).
        Most events are now handled internally by states.
        """
        self.logger.info("Handling external event: %s", event.value)
        await self.context.handle_event(event)


# --- Entry Point ---


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
