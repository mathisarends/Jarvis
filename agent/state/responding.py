import asyncio

from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext


class RespondingState(AssistantState):
    """State when generating and delivering response to user"""

    def __init__(self):
        super().__init__(StateType.RESPONDING)
        self._wake_word_task = None

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering Responding state - generating and delivering response"
        )

        context.ensure_realtime_audio_channel_paused()

        await self._start_wake_word_detection(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        await self._stop_wake_word_detection(context)

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED:
                self.logger.info(
                    "Assistant response completed - returning to waiting for user input"
                )
                await self._transition_to_timeout(context)
            case VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED:
                self.logger.info(
                    "Assistant speech interrupted - returning to listening"
                )
                await self._transition_to_listening(context)
            case VoiceAssistantEvent.WAKE_WORD_DETECTED:
                self.logger.info(
                    "Wake word detected during assistant response - interrupting and transitioning to listening"
                )
                await self._transition_to_listening(context)
            case VoiceAssistantEvent.IDLE_TRANSITION:
                self.logger.info("Idle transition in Responding state")
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to_error(context)
            case _:
                self.logger.debug("Ignoring event %s in Responding state", event.value)

    async def _start_wake_word_detection(self, context: VoiceAssistantContext) -> None:
        """Start wake word detection for interruption during assistant speech"""
        self.logger.debug("Starting wake word detection during assistant response")

        # Start wake word listening for interruption
        # WakeWordListener will publish WAKE_WORD_DETECTED directly via EventBus
        self._wake_word_task = asyncio.create_task(
            self._wake_word_detection_loop(context)
        )

    async def _stop_wake_word_detection(self, context: VoiceAssistantContext) -> None:
        """Stop wake word detection"""
        self.logger.debug("Stopping wake word detection")
        context.wake_word_listener.stop_listening()

        # Cancel the wake word detection task if it's running
        if self._wake_word_task and not self._wake_word_task.done():
            self._wake_word_task.cancel()
            try:
                await self._wake_word_task
            except asyncio.CancelledError:  # NOSONAR
                # Intentional cleanup - CancelledError not re-raised to avoid aborting state transition
                pass
            finally:
                self._wake_word_task = None

    async def _wake_word_detection_loop(self, context: VoiceAssistantContext) -> None:
        """Background loop for wake word detection"""
        try:
            wake_word_detected = await context.wake_word_listener.listen_for_wakeword()
            if wake_word_detected:
                # WakeWordListener publishes event directly via EventBus
                self.logger.info("Wake word detected during assistant response")
        except Exception as e:
            self.logger.error("Wake word detection error: %s", e)
            # Only publish error if EventBus is not available in WakeWordListener
            if not context.wake_word_listener.event_bus:
                context.event_bus.publish_sync(VoiceAssistantEvent.ERROR_OCCURRED)
