from __future__ import annotations

import asyncio
from typing import Any, Dict

from agent.realtime.event_bus import EventBus
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.tool import Tool
from agent.realtime.tools.views import FunctionCallItem, FunctionCallResult
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin
from agent.realtime.message_manager import RealtimeMessageManager


class ToolExecutor(LoggingMixin):
    """
    Executes tools based on function call events from the Realtime API.
    Supports both synchronous and long-running background tools.
    """

    def __init__(
        self, tool_registry: ToolRegistry, message_manager: RealtimeMessageManager
    ):
        self.tool_registry = tool_registry
        self.message_manager = message_manager
        self.event_bus = EventBus()
        
        # Track running background tasks
        self.background_tasks: Dict[str, asyncio.Task] = {}

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

            tool = self._retrieve_tool_from_registry(function_name)
            
            if tool.long_running:
                await self._handle_long_running_tool(tool, data)
            else:
                await self._handle_default_tool(tool, data)

        except Exception as e:
            self.logger.error(
                "Error handling tool call '%s': %s",
                data.name if data else "unknown",
                e,
                exc_info=True,
            )
            await self._send_error_result(data, str(e))

    async def _handle_default_tool(
        self, tool: Tool, data: FunctionCallItem
    ) -> None:
        """Handle synchronous tool execution."""
        try:
            self.logger.info("Executing synchronous tool: %s", tool.name)
            
            # Execute tool synchronously
            result = await tool.execute(data.arguments or {})
            
            self.logger.info(
                "Tool '%s' executed successfully with result: %s", tool.name, result
            )
            
            # Send result immediately
            await self._send_tool_result(data, result, tool.result_context)
            
        except Exception as e:
            self.logger.error(
                "Error executing synchronous tool '%s': %s", tool.name, e, exc_info=True
            )
            await self._send_error_result(data, str(e))

    async def _handle_long_running_tool(
        self, tool: Tool, data: FunctionCallItem
    ) -> None:
        """Handle long-running tool execution in background."""
        try:
            self.logger.info("Starting long-running tool: %s", tool.name)
            
            # Send loading message if available
            if tool.loading_message:
                await self.message_manager.send_loading_message_for_long_running_tool_call(
                    tool.loading_message
                )
                self.logger.info("Loading message sent for tool: %s", tool.name)
            
            # Start background execution
            task = asyncio.create_task(
                self._execute_long_running_tool_in_background(tool, data)
            )
            
            # Track the task
            task_id = f"{tool.name}_{data.call_id}"
            self.background_tasks[task_id] = task
            
            # Clean up task when done (don't await here!)
            task.add_done_callback(lambda t: self._cleanup_background_task(task_id))
            
            self.logger.info("Long-running tool '%s' started in background", tool.name)
            
        except Exception as e:
            self.logger.error(
                "Error starting long-running tool '%s': %s", tool.name, e, exc_info=True
            )
            await self._send_error_result(data, str(e))

    async def _execute_long_running_tool_in_background(
        self, tool: Tool, data: FunctionCallItem
    ) -> None:
        """Execute a long-running tool in the background and send result when done."""
        try:
            self.logger.info("Background execution started for tool: %s", tool.name)
            
            # Execute the tool
            result = await tool.execute(data.arguments or {})
            
            self.logger.info(
                "Long-running tool '%s' completed with result: %s", tool.name, result
            )
            
            # Send the result
            await self._send_tool_result(data, result, tool.result_context)
            
        except Exception as e:
            self.logger.error(
                "Error in background execution of tool '%s': %s", 
                tool.name, e, exc_info=True
            )
            await self._send_error_result(data, str(e))

    async def _send_tool_result(
        self, data: FunctionCallItem, result: Any, result_context: str = None
    ) -> None:
        """Send tool execution result."""
        try:
            function_call_result = FunctionCallResult(
                tool_name=data.name,
                call_id=data.call_id,
                output=result,
                result_context=result_context,
            )
            await self.message_manager.send_tool_result(function_call_result)
            self.logger.info("Tool result sent successfully for: %s", data.name)
            
        except Exception as e:
            self.logger.error(
                "Error sending tool result for '%s': %s", data.name, e, exc_info=True
            )

    async def _send_error_result(self, data: FunctionCallItem, error_message: str) -> None:
        """Send error result for failed tool execution."""
        try:
            function_call_result = FunctionCallResult(
                tool_name=data.name,
                call_id=data.call_id,
                output=f"Error: {error_message}",
                result_context="Tool execution failed",
            )
            await self.message_manager.send_tool_result(function_call_result)
            self.logger.info("Error result sent for tool: %s", data.name)
            
        except Exception as e:
            self.logger.error(
                "Error sending error result for '%s': %s", data.name, e, exc_info=True
            )

    def _retrieve_tool_from_registry(self, name: str) -> Tool:
        """Retrieve a tool from the registry."""
        tool = self.tool_registry.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry")
        return tool

    def _cleanup_background_task(self, task_id: str) -> None:
        """Clean up completed background task."""
        if task_id in self.background_tasks:
            task = self.background_tasks.pop(task_id)
            if task.exception():
                self.logger.error(
                    "Background task '%s' completed with exception: %s",
                    task_id, task.exception()
                )
            else:
                self.logger.info("Background task '%s' completed successfully", task_id)

    async def shutdown(self) -> None:
        """Gracefully shutdown the tool executor."""
        self.logger.info("Shutting down ToolExecutor...")
        
        # Cancel all running background tasks
        if self.background_tasks:
            self.logger.info("Cancelling %d background tasks", len(self.background_tasks))
            
            for task_id, task in self.background_tasks.items():
                if not task.done():
                    task.cancel()
                    self.logger.info("Cancelled background task: %s", task_id)
            
            # Wait for all tasks to complete/cancel
            if self.background_tasks:
                await asyncio.gather(
                    *self.background_tasks.values(), 
                    return_exceptions=True
                )
            
            self.background_tasks.clear()
        
        self.logger.info("ToolExecutor shutdown complete")

    def get_running_tasks_count(self) -> int:
        """Get the number of currently running background tasks."""
        return len([task for task in self.background_tasks.values() if not task.done()])

    def get_running_task_names(self) -> list[str]:
        """Get the names of currently running background tasks."""
        return [
            task_id for task_id, task in self.background_tasks.items() 
            if not task.done()
        ]