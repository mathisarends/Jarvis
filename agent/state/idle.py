from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

from agent.state.base import AssistantState, StateType, VoiceAssistantEvent

if TYPE_CHECKING:
    from agent.state.base import VoiceAssistantContext


class IdleState(AssistantState):
    """Initial state - waiting for wake word"""

    def __init__(self):
        super().__init__(StateType.IDLE)
        self._wake_task: Optional[asyncio.Task] = None

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Idle state - starting wake word detection")
        context.end_session()

        # Start wake word detection
        await self._start_wake_word_detection(context)

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Exiting Idle state - stopping wake word detection")
        await self._stop_wake_word_detection()

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.WAKE_WORD_DETECTED:
                context.start_session()
                await self._transition_to_timeout(context)
            case _:
                self.logger.debug("Ignoring event %s in Idle state", event.value)

    async def _start_wake_word_detection(self, context: VoiceAssistantContext) -> None:
        """Start the wake word detection task"""
        if self._wake_task and not self._wake_task.done():
            self.logger.debug("Wake word task already running")
            return

        self.logger.debug("Starting wake word detection task")
        self._wake_task = asyncio.create_task(
            self._wake_word_loop(context), name="wake_word_detection"
        )

    async def _stop_wake_word_detection(self) -> None:
        """Stop the wake word detection task"""
        if not self._wake_task or self._wake_task.done():
            self.logger.debug("No wake word task to stop or already done")
            return

        self.logger.debug("Stopping wake word detection task")
        self._wake_task.cancel()
        try:
            await self._wake_task
        except asyncio.CancelledError:  # NOSONAR
            self.logger.debug("Wake word task cancelled")
        except Exception:
            self.logger.exception("Error while stopping wake word task")
        finally:
            self._wake_task = None

    async def _wake_word_loop(self, context: VoiceAssistantContext) -> None:
        """Wake word detection loop - runs until cancelled"""
        try:
            while True:
                detected = await context.wake_word_listener.listen_for_wakeword()
                if detected:
                    self.logger.info("Wake word detected!")
                    # WakeWordListener now publishes event directly via EventBus
                    break  # Exit loop after detection

                # Small delay to prevent busy loop
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:  # NOSONAR
            self.logger.debug("Wake word loop cancelled")
            raise
        except Exception:
            self.logger.exception("Wake word detection failed")
            # Only publish error if EventBus is not available in WakeWordListener
            if not context.wake_word_listener.event_bus:
                context.event_bus.publish_sync(VoiceAssistantEvent.ERROR_OCCURRED)
