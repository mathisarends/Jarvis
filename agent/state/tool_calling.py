from agent.state.base import AssistantState, StateType, VoiceAssistantEvent
from agent.state.context import VoiceAssistantContext


# This state exists exists because long running tool calls can lead to timeouts of the agent in responding state
class ToolCallingState(AssistantState):
    """State when executing tool calls requested by the assistant"""

    def __init__(self):
        super().__init__(StateType.TOOL_CALLING)

    async def handle(
        self, event: VoiceAssistantEvent, context: VoiceAssistantContext
    ) -> None:
        """Handle events in tool calling state"""
        match event:
            case VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT:
                self.logger.info("Tool call result received")
                await self._transition_to_responding(context)
            case VoiceAssistantEvent.ASSISTANT_COMPLETED_MCP_TOOL_CALL_RESULT:
                self.logger.info("MCP tool call completed - transitioning to Responding state")
                await self._transition_to_responding(context)
            case VoiceAssistantEvent.ASSISTANT_FAILED_MCP_TOOL_CALL:
                self.logger.info("MCP tool call failed - transitioning to Responding state")
                await self._transition_to_responding(context)
            case _:
                self.logger.debug(
                    "Ignoring event %s in Tool Calling state", event.value
                )
