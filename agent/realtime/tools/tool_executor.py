from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Generator
import inspect

from agent.realtime.event_bus import EventBus
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.tool import Tool
from agent.realtime.tools.views import FunctionCallItem, FunctionCallResult
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin
from agent.realtime.messaging.message_manager import RealtimeMessageManager


class ToolExecutor(LoggingMixin):
    """
    Executes tools based on function call events from the Realtime API.
    """

    def __init__(
        self, tool_registry: ToolRegistry, message_manager: RealtimeMessageManager
    ):
        self.tool_registry = tool_registry
        self.message_manager = message_manager
        self.event_bus = EventBus()
        self._background_tasks = set()  # Keep track of background tasks

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

            await self._execute_tool(tool, data)

        except Exception as e:
            await self._handle_tool_error(data, e, "handling tool call")

    async def _execute_tool(self, tool: Tool, data: FunctionCallItem) -> None:
        """Execute a tool synchronously or as background task for generators."""
        self.logger.info("Executing tool: %s", tool.name)

        if tool.is_generator:
            task = asyncio.create_task(self._execute_generator_tool(tool, data))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            # Execute regular tools synchronously
            result = await tool.execute(data.arguments or {})
            self.logger.info("Tool '%s' executed successfully", tool.name)
            await self._send_tool_result(data, result, tool.result_context)

    async def _send_tool_result(
        self, data: FunctionCallItem, result: Any, result_context: str = None
    ) -> None:
        """Send tool execution result."""
        function_call_result = FunctionCallResult(
            tool_name=data.name,
            call_id=data.call_id,
            output=result,
            result_context=result_context,
        )
        await self.message_manager.send_tool_result(function_call_result)
        self.logger.info("Tool result sent for: %s", data.name)

        await self._check_and_publish_all_tools_finished()

    def _retrieve_tool_from_registry(self, name: str) -> Tool:
        """Retrieve a tool from the registry."""
        tool = self.tool_registry.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry")
        return tool

    async def _check_and_publish_all_tools_finished(self) -> None:
        """Check if all tools are finished and publish event if so."""
        self.logger.info(
            "Tool finished - publishing ASSISTANT_RECEIVED_TOOL_CALL_RESULT event"
        )
        await self.event_bus.publish_async(
            VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT
        )

    async def _execute_generator_tool(self, tool: Tool, data: FunctionCallItem) -> None:
        """Execute a generator tool in the background with status updates."""
        try:
            self.logger.info("Starting generator tool execution: %s", tool.name)

            result = await tool.execute(data.arguments or {})

            # Early return if not a generator
            if not (inspect.isgenerator(result) or inspect.isasyncgen(result)):
                await self._send_tool_result(data, result, tool.result_context)
                return

            if inspect.isasyncgen(result):
                await self._send_async_generator_updates(result)
            else:
                await self._send_sync_generator_updates(result)

            self.logger.info("Generator tool '%s' completed successfully", tool.name)

        except Exception as e:
            await self._handle_tool_error(
                data, e, f"executing generator tool '{tool.name}'"
            )

    async def _send_async_generator_updates(self, generator) -> None:
        """Send updates for an async generator."""
        async for chunk in generator:
            await self.message_manager.send_update_for_generator_tool(chunk)
            self.logger.debug("Generator yielded: %s", chunk)

    async def _send_sync_generator_updates(self, generator) -> None:
        """Send updates for a sync generator."""
        for chunk in generator:
            await self.message_manager.send_update_for_generator_tool(chunk)
            self.logger.debug("Generator yielded: %s", chunk)

    async def _handle_tool_error(
        self, data: FunctionCallItem, error: Exception, context: str
    ) -> None:
        """Handle tool execution errors."""
        error_message = f"Error {context}: {str(error)}"
        self.logger.error(error_message, exc_info=True)

        # Send error result
        function_call_result = FunctionCallResult(
            tool_name=data.name,
            call_id=data.call_id,
            output=f"Error: {str(error)}",
            result_context="This is an error message that should be communicated to the user",
        )
        await self.message_manager.send_tool_result(function_call_result)

        # Still publish the finished event to prevent hanging
        await self._check_and_publish_all_tools_finished()
