from __future__ import annotations

from typing import Any

from agent.realtime.event_bus import EventBus
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.views import FunctionCallItem, FunctionCallResult
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


from agent.realtime.message_manager import RealtimeMessageManager


class ToolExecutor(LoggingMixin):
    """
    Executes tools based on function call events from the Realtime API.
    Subscribes to ASSISTANT_STARTED_TOOL_CALL events and executes the requested tools.
    """

    def __init__(self, tool_registry: ToolRegistry, message_manager: RealtimeMessageManager):
        super().__init__()
        self.tool_registry = tool_registry
        self.message_manager = message_manager
        self.event_bus = EventBus()

        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_STARTED_TOOL_CALL, self._handle_tool_call
        )
        self.logger.info("ToolExecutor initialized and subscribed to tool call events")

    async def _handle_tool_call(
        self, event: VoiceAssistantEvent, data: FunctionCallItem
    ) -> None:
        """Handle function call events and execute the requested tool."""
        try:
            function_name = data.name
            arguments = data.arguments or {}

            self.logger.info(
                "Executing tool: %s with arguments: %s", function_name, arguments
            )

            # Execute tool and get result
            result = await self._execute_tool(function_name, arguments)

            self.logger.info(
                "Tool '%s' executed successfully with result: %s", function_name, result
            )
            self.logger.info(f"Tool Result ({function_name}): {result}")

            function_call_result = FunctionCallResult(
                tool_name=data.name, call_id=data.call_id, output=result
            )
            await self.message_manager.send_tool_result(function_call_result)

        except Exception as e:
            self.logger.error(
                "Error executing tool '%s': %s",
                data.name if data else "unknown",
                e,
                exc_info=True,
            )
            print(f"Tool Execution Error: {e}")

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        tool = self.tool_registry.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        return await tool.execute(arguments)
