from __future__ import annotations

import asyncio
import inspect
from typing import Any, get_type_hints

from agent.config.views import AgentConfig
from agent.realtime.event_bus import EventBus
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.tool import Tool
from agent.realtime.tools.views import (
    FunctionCallItem,
    FunctionCallResult,
    SpecialToolParameters,
)
from agent.state.base import VoiceAssistantEvent
from audio.player.audio_manager import AudioManager
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
        audio_manager: AudioManager,
        agent_config: AgentConfig
    ):
        self.tool_registry = tool_registry
        self.message_manager = message_manager
        self.audio_manager = audio_manager
        self.agent_config = agent_config
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
            llm_arguments = data.arguments or {}

            self.logger.info(
                "Executing tool: %s with LLM arguments: %s",
                function_name,
                llm_arguments,
            )

            tool = self._retrieve_tool_from_registry(function_name)

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

        if tool.is_generator:
            task = asyncio.create_task(
                self._execute_generator_tool(tool, data, final_arguments)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            # Execute regular tools synchronously
            result = await tool.execute(final_arguments)
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

    async def _execute_generator_tool(
        self, tool: Tool, data: FunctionCallItem, final_arguments: dict[str, Any]
    ) -> None:
        """Execute a generator tool in the background with status updates."""
        try:
            self.logger.info("Starting generator tool execution: %s", tool.name)

            result = await tool.execute(final_arguments)

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

    def _inject_special_parameters(
        self, func: callable, llm_arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Inject special parameters into function arguments if the function expects them.

        Args:
            func: The tool function to analyze
            llm_arguments: Arguments provided by the LLM

        Returns:
            Combined arguments with injected special parameters
        """
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Start with LLM arguments
        final_arguments = llm_arguments.copy()

        # Get special parameter names from SpecialToolParameters
        special_param_names = set(SpecialToolParameters.model_fields.keys())

        for param_name, param in signature.parameters.items():
            print("param", param_name, param)
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
        special_params_map = {
            "audio_manager": self.audio_manager,
            "event_bus": self.event_bus,
            "agent_config": self.agent_config,
            # Hier können weitere special parameters hinzugefügt werden:
            # 'message_manager': self.message_manager,
            # 'event_bus': self.event_bus,
            # 'tool_registry': self.tool_registry,
        }

        return special_params_map.get(param_name)
