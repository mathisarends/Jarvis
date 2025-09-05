from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext


# This state exists exists because long running tool calls can lead to timeouts of the agent in responding state
class ToolCallingState(AssistantState):
    """State when executing tool calls requested by the assistant"""

    def __init__(self):
        super().__init__(StateType.TOOL_CALLING)

    async def on_enter(self, context: VoiceAssistantContext) -> None:
        """Called when entering tool calling state"""
        self.logger.info("Entering Tool Calling state")

    async def on_exit(self, context: VoiceAssistantContext) -> None:
        """Called when exiting tool calling state"""
        self.logger.info("Exiting Tool Calling state")

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        """Handle events in tool calling state"""
        match event:
            case VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT:
                self.logger.info("Tool call result received")
                await self._transition_to_responding(context)
            case _:
                self.logger.debug(
                    "Ignoring event %s in Tool Calling state", event.value
                )
