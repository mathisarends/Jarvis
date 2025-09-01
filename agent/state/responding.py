from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext


class RespondingState(AssistantState):
    """State when generating and delivering response to user"""

    def __init__(self):
        super().__init__(StateType.RESPONDING)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        self.logger.info(
            "Entering Responding state - generating and delivering response"
        )

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        # Nothing to clean up in responding state
        pass

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        match event:
            case VoiceAssistantEvent.ASSISTANT_STARTED_RESPONDING:
                self.logger.info("Assistant started responding")
                # Stay in responding state
            case VoiceAssistantEvent.ASSISTANT_STARTED_TOOL_CALL:
                self.logger.info("Assistant started tool call")
                # Stay in responding state during tool calls
            case VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT:
                self.logger.info("Assistant completed tool call")
                # Stay in responding state - might have more to say
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
            case VoiceAssistantEvent.IDLE_TRANSITION:
                self.logger.info("Idle transition in Responding state")
                await self._transition_to_idle(context)
            case VoiceAssistantEvent.ERROR_OCCURRED:
                await self._transition_to_error(context)
            case _:
                self.logger.debug("Ignoring event %s in Responding state", event.value)
