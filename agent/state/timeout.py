import asyncio
from typing import Optional
from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext
from audio.file_player import SoundFile


class TimeoutState(AssistantState):
    """State after wake word - waiting for user to start speaking with timeout"""

    TIMEOUT_SECONDS = 10.0

    def __init__(self):
        super().__init__(StateType.TIMEOUT)
        self._timeout_task: Optional[asyncio.Task] = None

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering TimeoutState - user has %s seconds to start speaking",
            self.TIMEOUT_SECONDS,
        )

        # Play wake-word sound cue
        context.sound_player.play_sound_file(SoundFile.WAKE_WORD)

        # Start timeout
        await self._start_timeout(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._cancel_timeout()

    async def _start_timeout(self, context: VoiceAssistantContext) -> None:
        """Start the timeout task for this state"""
        await self._cancel_timeout()  # Cancel any existing timeout

        self.logger.debug("Starting timeout: %s seconds", self.TIMEOUT_SECONDS)
        self._timeout_task = asyncio.create_task(self._timeout_handler(context))

    async def _timeout_handler(self, context: VoiceAssistantContext) -> None:
        """Handle timeout by sending TIMEOUT_OCCURRED event"""
        try:
            await asyncio.sleep(self.TIMEOUT_SECONDS)
            self.logger.info("Timeout occurred after %s seconds", self.TIMEOUT_SECONDS)
            await context.handle_event(VoiceAssistantEvent.TIMEOUT_OCCURRED)
        except asyncio.CancelledError:
            self.logger.debug("Timeout cancelled")

    async def _cancel_timeout(self) -> None:
        """Cancel any running timeout"""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
            self._timeout_task = None

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.USER_STARTED_SPEAKING:
                self.logger.info("User started speaking - cancelling timeout")
                await self._transition_to_listening(context)
            case VoiceAssistantEvent.TIMEOUT_OCCURRED:
                self.logger.info(
                    "Timeout occurred - user did not start speaking within %s seconds",
                    self.TIMEOUT_SECONDS,
                )
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.IDLE_TRANSITION:
                self.logger.info("Idle transition in WaitingForUserInput state")
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to_error(context)
            case _:
                self.logger.debug(
                    "Ignoring event %s in WaitingForUserInput state", event.value
                )
