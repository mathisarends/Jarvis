from __future__ import annotations

import asyncio
import inspect
from typing import Any

from agent.realtime.event_bus import EventBus
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.tool import Tool
from agent.realtime.tools.views import (
    FunctionCallItem,
    FunctionCallResult,
    SpecialToolParameters,
)
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin
from agent.realtime.messaging.message_manager import RealtimeMessageManager


class ToolExecutor(LoggingMixin):
    """
    Executes tools based on function call events from the Realtime API.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        message_manager: RealtimeMessageManager,
        special_tool_parameters: SpecialToolParameters,
        event_bus: EventBus,
    ):
        self.tool_registry = tool_registry
        self.message_manager = message_manager
        self.special_tool_parameters = special_tool_parameters
        self._background_tasks = set()

        # Only extract what's used directly
        self.event_bus = event_bus

        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_STARTED_TOOL_CALL, self._handle_tool_call
        )
        self.logger.info("ToolExecutor initialized and subscribed to tool call events")

    async def _handle_tool_call(self, data: FunctionCallItem) -> None:
        """Handle function call events and execute the requested tool."""
        try:
            function_name = data.name
            llm_arguments = data.arguments or {}

            self.logger.info(
                "Executing tool: %s with LLM arguments: %s",
                function_name,
                llm_arguments,
            )

            tool = self._retrieve_tool_from_registry(function_name)

            if tool.execution_message:
                await self.message_manager.send_execution_message(
                    tool.execution_message
                )

            await self._execute_tool(tool, data, llm_arguments)

        except Exception as e:
            await self._handle_tool_error(data, e, "handling tool call")

    async def _execute_tool(
        self, tool: Tool, data: FunctionCallItem, llm_arguments: dict[str, Any]
    ) -> None:
        """Execute a tool synchronously or as background task for generators."""
        self.logger.info("Executing tool: %s", tool.name)

        try:
            # Inject special parameters
            final_arguments = self._inject_special_parameters(
                tool.function, llm_arguments
            )
            self.logger.debug(
                "Final arguments after injection: %s", list(final_arguments.keys())
            )

        except Exception as e:
            await self._handle_tool_error(
                data, e, f"injecting parameters for tool '{tool.name}'"
            )
            return

        if tool.is_async_generator:
            task = asyncio.create_task(
                self._execute_generator_tool(tool, data, final_arguments)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            # Execute regular tools synchronously
            result = await tool.execute(final_arguments)
            self.logger.info("Tool '%s' executed successfully", tool.name)
            await self._send_tool_result(data, result, tool.response_instruction)

    async def _send_tool_result(
        self, data: FunctionCallItem, result: Any, response_instruction: str = None
    ) -> None:
        """Send tool execution result."""
        function_call_result = FunctionCallResult(
            tool_name=data.name,
            call_id=data.call_id,
            output=result,
            response_instruction=response_instruction,
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

    async def _execute_generator_tool(
        self, tool: Tool, data: FunctionCallItem, final_arguments: dict[str, Any]
    ) -> None:
        """Execute a generator tool in the background with status updates."""
        try:
            self.logger.info("Starting generator tool execution: %s", tool.name)

            result = await tool.execute(final_arguments)

            # Early return if not a generator
            if not (inspect.isgenerator(result) or inspect.isasyncgen(result)):
                await self._send_tool_result(data, result, tool.response_instruction)
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
            await self.message_manager.send_execution_message(chunk)
            self.logger.debug("Generator yielded: %s", chunk)

    async def _send_sync_generator_updates(self, generator) -> None:
        """Send updates for a sync generator."""
        for chunk in generator:
            await self.message_manager.send_execution_message(chunk)
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
            response_instruction="This is an error message that should be communicated to the user",
        )
        await self.message_manager.send_tool_result(function_call_result)

        # Still publish the finished event to prevent hanging
        await self._check_and_publish_all_tools_finished()

    def _inject_special_parameters(
        self, func: callable, llm_arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Inject special parameters into function arguments if the function expects them.
        """
        signature = inspect.signature(func)

        # Start with LLM arguments
        final_arguments = llm_arguments.copy()

        # Get special parameter names from SpecialToolParameters
        special_param_names = set(SpecialToolParameters.model_fields.keys())

        for param_name, param in signature.parameters.items():
            # Skip if already provided by LLM
            if param_name in final_arguments:
                continue

            # Skip 'self'/'cls'
            if param_name in ("self", "cls"):
                continue

            # Check if this parameter is a special parameter
            if param_name in special_param_names:
                injected_value = self._get_special_parameter_value(param_name)

                if injected_value is not None:
                    final_arguments[param_name] = injected_value
                    self.logger.debug(
                        f"Injected special parameter '{param_name}' for tool function"
                    )
                elif param.default == inspect.Parameter.empty:
                    # Required parameter but no value available
                    raise ValueError(
                        f"Required special parameter '{param_name}' is not available"
                    )

        return final_arguments

    def _get_special_parameter_value(self, param_name: str) -> Any:
        """Get the value for a special parameter by name."""
        return getattr(self.special_tool_parameters, param_name, None)
