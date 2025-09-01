"""
Efficient event-driven controller for voice assistant
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from agent.realtime.views import AssistantVoice, RealtimeModel
from agent.state.base import (
    VoiceAssistantContext,
    VoiceAssistantEvent,
)
from audio import SoundFilePlayer, SoundFile
from audio.wake_word_listener import WakeWordListener, PorcupineBuiltinKeyword
from shared.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class VoiceAssistantConfig:
    """Configuration for the voice assistant."""

    voice: AssistantVoice = AssistantVoice.ALLOY
    realtime_model: RealtimeModel = RealtimeModel.GPT_REALTIME

    wake_word: PorcupineBuiltinKeyword = PorcupineBuiltinKeyword.PICOVOICE
    sensitivity: float = 0.7


class VoiceAssistantController(LoggingMixin):
    """Event-driven controller for voice assistant state machine"""

    def __init__(self):
        self.context = VoiceAssistantContext()
        self.sound_player = SoundFilePlayer()
        self.wake_word_listener = WakeWordListener(
            wakeword=PorcupineBuiltinKeyword.PICOVOICE, sensitivity=0.7
        )

        self.event_sounds: dict[VoiceAssistantEvent, SoundFile] = {
            VoiceAssistantEvent.WAKE_WORD_DETECTED: SoundFile.WAKE_WORD,
            VoiceAssistantEvent.SESSION_TIMEOUT: SoundFile.RETURN_TO_IDLE,
            VoiceAssistantEvent.ERROR_OCCURRED: SoundFile.ERROR,
        }

        self._running = False
        self._current_tasks: dict[str, Optional[asyncio.Task]] = {
            "wake_word": None,
            "session_timeout": None,
            "user_input": None,
        }

        self.logger.info("Voice Assistant Controller initialized")

    async def start(self) -> None:
        """Start the voice assistant"""
        if self._running:
            self.logger.warning("Controller is already running")
            return

        self._running = True
        self.logger.info("Starting Voice Assistant Controller")

        self.sound_player.play_startup_sound()

        try:
            await self._main_loop()
        except Exception as e:
            self.logger.error("Error in main loop: %s", e)
            await self.handle_event(VoiceAssistantEvent.ERROR_OCCURRED)
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the voice assistant"""
        if not self._running:
            return

        self.logger.info("Stopping Voice Assistant Controller")
        self._running = False

        # Cancel all running tasks
        await self._cancel_all_tasks()

        # Cleanup services
        self.wake_word_listener.cleanup()
        self.sound_player.stop_sound()

        self.logger.info("Voice Assistant Controller stopped")

    async def handle_event(self, event: VoiceAssistantEvent) -> None:
        """Handle an event with sound feedback and state transition"""
        self.logger.info("Handling event: %s", event.value)

        # Play sound for event (non-blocking)
        if event in self.event_sounds:
            sound_file = self.event_sounds[event]
            self.sound_player.play_sound_file(sound_file)
            self.logger.debug(
                "Playing sound for event %s: %s", event.value, sound_file.value
            )

        # Handle state transition
        self.context.handle_event(event)

        # Update tasks based on new state
        await self._update_tasks_for_current_state()

    async def _main_loop(self) -> None:
        """Main event-driven control loop"""
        # Start initial tasks based on current state
        await self._update_tasks_for_current_state()

        # Keep running until stopped
        while self._running:
            # Wait for any task to complete
            if any(task for task in self._current_tasks.values() if task):
                done, pending = await asyncio.wait(
                    [task for task in self._current_tasks.values() if task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Process completed tasks
                for task in done:
                    await self._handle_completed_task(task)
            else:
                # No tasks running, wait briefly to avoid busy loop
                await asyncio.sleep(0.1)

    async def _update_tasks_for_current_state(self) -> None:
        """Update running tasks based on current state"""
        current_state_name = type(self.context.state).__name__
        self.logger.debug("Updating tasks for state: %s", current_state_name)

        # Cancel tasks that shouldn't run in current state
        await self._cancel_inappropriate_tasks(current_state_name)

        # Start tasks needed for current state
        if current_state_name == "IdleState":
            await self._start_wake_word_task()

        elif current_state_name == "ListeningState":
            await self._start_session_timeout_task()
            await self._start_user_input_task()

        elif current_state_name == "RespondingState":
            await self._start_response_task()
            await self._start_session_timeout_task()

        elif current_state_name == "ErrorState":
            await self._start_error_recovery_task()

    async def _start_wake_word_task(self) -> None:
        """Start wake word detection task"""
        if not self._current_tasks["wake_word"]:
            self.logger.debug("Starting wake word detection task")
            self._current_tasks["wake_word"] = asyncio.create_task(
                self._wake_word_detection(), name="wake_word_detection"
            )

    async def _start_session_timeout_task(self) -> None:
        """Start session timeout task"""
        if not self._current_tasks["session_timeout"]:
            self.logger.debug("Starting session timeout task")
            self._current_tasks["session_timeout"] = asyncio.create_task(
                self._session_timeout(), name="session_timeout"
            )

    async def _start_user_input_task(self) -> None:
        """Start user input detection task"""
        if not self._current_tasks["user_input"]:
            self.logger.debug("Starting user input detection task")
            self._current_tasks["user_input"] = asyncio.create_task(
                self._user_input_detection(), name="user_input_detection"
            )

    async def _start_response_task(self) -> None:
        """Start response generation and playback task"""
        if not self._current_tasks["user_input"]:  # Reuse user_input slot for response
            self.logger.debug("Starting response task")
            self._current_tasks["user_input"] = asyncio.create_task(
                self._generate_and_play_response(), name="response_generation"
            )

    async def _start_error_recovery_task(self) -> None:
        """Start error recovery task"""
        if not self._current_tasks["user_input"]:  # Reuse slot
            self.logger.debug("Starting error recovery task")
            self._current_tasks["user_input"] = asyncio.create_task(
                self._error_recovery(), name="error_recovery"
            )

    async def _cancel_inappropriate_tasks(self, current_state: str) -> None:
        """Cancel tasks that shouldn't run in the current state"""
        tasks_to_cancel = []

        if current_state != "IdleState" and self._current_tasks["wake_word"]:
            tasks_to_cancel.append("wake_word")

        if (
            current_state not in ["ListeningState", "RespondingState"]
            and self._current_tasks["session_timeout"]
        ):
            tasks_to_cancel.append("session_timeout")

        for task_name in tasks_to_cancel:
            if self._current_tasks[task_name]:
                self.logger.debug("Cancelling task: %s", task_name)
                self._current_tasks[task_name].cancel()
                try:
                    await self._current_tasks[task_name]
                except asyncio.CancelledError:
                    pass
                self._current_tasks[task_name] = None

    async def _cancel_all_tasks(self) -> None:
        """Cancel all running tasks"""
        for task_name, task in self._current_tasks.items():
            if task and not task.done():
                self.logger.debug("Cancelling task: %s", task_name)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Clear all tasks
        for key in self._current_tasks:
            self._current_tasks[key] = None

    async def _handle_completed_task(self, completed_task: asyncio.Task) -> None:
        """Handle a completed task"""
        task_name = completed_task.get_name()

        try:
            result = await completed_task
            self.logger.debug("Task %s completed with result: %s", task_name, result)

            # Clear the completed task
            for key, task in self._current_tasks.items():
                if task == completed_task:
                    self._current_tasks[key] = None
                    break

        except asyncio.CancelledError:
            self.logger.debug("Task %s was cancelled", task_name)
        except Exception as e:
            self.logger.error("Task %s failed with error: %s", task_name, e)
            await self.handle_event(VoiceAssistantEvent.ERROR_OCCURRED)

    # Task implementations
    async def _wake_word_detection(self) -> bool:
        """Detect wake word"""
        detected = await self.wake_word_listener.listen_for_wakeword()
        if detected:
            await self.handle_event(VoiceAssistantEvent.WAKE_WORD_DETECTED)

            # 🔄 Demo: switch to "bumblebee" for next iteration
            self.logger.info("Switching wake word to BUMBLEBEE for demo...")
            self.wake_word_listener.set_wakeword(PorcupineBuiltinKeyword.BUMBLEBEE)

            return True
        return False

    async def _session_timeout(self) -> None:
        """Handle session timeout"""
        await asyncio.sleep(20.0)  # 20 second timeout
        await self.handle_event(VoiceAssistantEvent.SESSION_TIMEOUT)

    async def _user_input_detection(self) -> None:
        """Detect user input (placeholder)"""
        # TODO: Implement actual speech recognition
        self.logger.info("Listening for user input...")
        await asyncio.sleep(5.0)  # Simulate listening

        # Simulate receiving user input
        await self.handle_event(VoiceAssistantEvent.USER_INPUT_RECEIVED)

    async def _generate_and_play_response(self) -> None:
        """Generate and play response (placeholder)"""
        # TODO: Implement actual LLM call and TTS
        self.logger.info("Generating and playing response...")
        await asyncio.sleep(3.0)  # Simulate response generation and playback

        await self.handle_event(VoiceAssistantEvent.SPEECH_DONE)

    async def _error_recovery(self) -> None:
        """Handle error recovery"""
        self.logger.info("Performing error recovery...")
        await asyncio.sleep(2.0)  # Give time for error sound to play

        await self.handle_event(VoiceAssistantEvent.SPEECH_DONE)


async def main():
    """Main entry point"""
    controller = VoiceAssistantController()

    try:
        await controller.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await controller.stop()


if __name__ == "__main__":
    asyncio.run(main())
