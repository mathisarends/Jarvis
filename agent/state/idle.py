from __future__ import annotations

from typing import TYPE_CHECKING

from agent.state.base import AssistantState, StateType, VoiceAssistantEvent

if TYPE_CHECKING:
    from agent.state.base import VoiceAssistantContext


class IdleState(AssistantState):
    """Initial state - waiting for wake word"""

    def __init__(self):
        super().__init__(StateType.IDLE)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info("Entering Idle state - waiting for wake word")
        context.end_session()

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        # TODO: Cleanup wake word listener here and also call this method here 
        pass

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.WAKE_WORD_DETECTED:
                context.start_session()
                # Import hier statt oben - vermeidet circular import
                from agent.state.timeout import TimeoutState
                await self._transition_to(TimeoutState(), context)
            case _:
                self.logger.debug("Ignoring event %s in Idle state", event.value)