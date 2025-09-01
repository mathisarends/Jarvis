"""
Minimal controller: only wake-word event supported
"""
import asyncio
from dataclasses import dataclass
from typing import Optional

from agent.realtime.views import AssistantVoice, RealtimeModel
from agent.state.base import VoiceAssistantContext, VoiceAssistantEvent, StateType
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
    """Slim controller: boot, wake-word, and state transition (only)."""

    def __init__(self, config: Optional[VoiceAssistantConfig] = None):
        self.config = config or VoiceAssistantConfig()

        self.sound_player = SoundFilePlayer()
        self.context = VoiceAssistantContext(sound_player=self.sound_player)
        self.wake_word_listener = WakeWordListener(
            wakeword=self.config.wake_word,
            sensitivity=self.config.sensitivity,
        )

        # nur ein Task in dieser Minimalversion
        self._wake_task: Optional[asyncio.Task] = None
        self._running = False

        self.logger.info("Voice Assistant Controller initialized (minimal wake-word mode)")

    async def start(self) -> None:
        if self._running:
            self.logger.warning("Controller already running")
            return

        self._running = True
        self.logger.info("Starting Voice Assistant Controller")
        self.sound_player.play_startup_sound()

        # Initialize the initial state
        await self.context.state.on_enter(self.context)

        try:
            # initial: je nach State entscheiden, ob wake-word laufen soll
            await self._update_tasks_for_state()
            # idle loop – läuft bis stop() gerufen wird
            while self._running:
                await asyncio.sleep(0.1)
        except Exception:
            self.logger.exception("Unhandled error in controller")
            # In der Minimalversion haben wir nur WAKE; Error-State könntest du später anbinden.
        finally:
            await self.stop()

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.logger.info("Stopping Voice Assistant Controller")

        # Task beenden
        await self._cancel_wake_task()

        # Services aufräumen
        try:
            self.wake_word_listener.cleanup()
        except Exception:
            self.logger.exception("WakeWord cleanup failed")

        try:
            self.sound_player.stop_sound()
        except Exception:
            self.logger.exception("Sound stop failed")

        self.logger.info("Voice Assistant Controller stopped")

    async def handle_event(self, event: VoiceAssistantEvent) -> None:
        """Only handles WAKE_WORD_DETECTED in this minimal setup."""
        self.logger.info("Handling event: %s", event.value)

        if event is VoiceAssistantEvent.WAKE_WORD_DETECTED:
            # State-Transition (Idle -> Listening) - sound will be played in state
            await self.context.handle_event(event)

            # Tasks gemäß neuem State anpassen
            await self._update_tasks_for_state()
        else:
            # alle anderen Events werden hier (noch) ignoriert
            self.logger.debug("Ignoring event in minimal mode: %s", event.value)

    # --- intern ---------------------------------------------------------------

    async def _update_tasks_for_state(self) -> None:
        """In Minimalversion: Wake-Word nur in Idle laufen lassen."""
        current_state_type = self.context.state.state_type
        self.logger.debug("Update tasks for state: %s", current_state_type.value)

        if current_state_type == StateType.IDLE:
            await self._start_wake_task()
        else:
            # ListeningState / RespondingState / ErrorState -> Wake-Word stoppen
            await self._cancel_wake_task()

    async def _start_wake_task(self) -> None:
        if self._wake_task and not self._wake_task.done():
            return  
        self.logger.debug("Starting wake-word detection task")
        self._wake_task = asyncio.create_task(self._wake_word_loop(), name="wake_word")

    async def _cancel_wake_task(self) -> None:
        if not self._wake_task or self._wake_task.done():
            self.logger.debug("No wake task to cancel or already done")
            return
        
        self.logger.debug("Cancelling wake-word task")
        self._wake_task.cancel()
        try:
            await self._wake_task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("Error while cancelling wake task")
        finally:
            self._wake_task = None

    async def _wake_word_loop(self) -> None:
        """Wartet einmalig auf Wake-Word und feuert dann das Event."""
        try:
            detected = await self.wake_word_listener.listen_for_wakeword()
            if detected:
                await self.handle_event(VoiceAssistantEvent.WAKE_WORD_DETECTED)

        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("Wake-word loop failed")
            # TODO: Add tranistion to error state here


# --- optionaler Entry-Point ---------------------------------------------------

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