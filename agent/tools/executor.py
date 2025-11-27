from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any

from agent.events import EventBus
from agent.state.base import VoiceAssistantEvent
from agent.tools.models import (
    FunctionCallItem,
    FunctionCallResult,
    SpecialToolParameters,
)
from agent.tools.registry.models import Tool
from agent.tools.registry.service import ToolRegistry
from shared.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from agent.realtime.messaging.message_manager import RealtimeMessageManager


class ToolExecutor(LoggingMixin):
    def __init__(
        self,
        tool_registry: ToolRegistry,
        message_manager: RealtimeMessageManager,
        special_tool_parameters: SpecialToolParameters,
        event_bus: EventBus,
    ):
        self._tool_registry = tool_registry
        self._message_manager = message_manager
        self._special_tool_parameters = special_tool_parameters
        self._background_tasks: set[asyncio.Task] = set()
        self._event_bus = event_bus

        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_STARTED_TOOL_CALL, self._handle_tool_call
        )
        self.logger.info("ToolExecutor initialized and subscribed to tool call events")

    async def _handle_tool_call(self, data: FunctionCallItem) -> None:
        function_name = data.name
        llm_arguments = data.arguments or {}

        self.logger.info(
            "Executing tool: %s with arguments: %s",
            function_name,
            llm_arguments,
        )

        tool = self._tool_registry.get(function_name)

        if tool.execution_message:
            await self._message_manager.send_execution_message(tool.execution_message)

        await self._execute_tool(tool, data, llm_arguments)

    async def _execute_tool(
        self, tool: Tool, data: FunctionCallItem, llm_arguments: dict[str, Any]
    ) -> None:
        self.logger.info("Executing tool: %s", tool.name)

        final_arguments = self._build_final_arguments(tool.function, llm_arguments)

        if tool.is_async_generator:
            self._execute_async_generator_in_background(tool, data, final_arguments)
        else:
            result = await tool.execute(final_arguments)
            self.logger.info("Tool '%s' executed successfully", tool.name)
            await self._send_result_and_notify(data, result, tool.response_instruction)

    def _execute_async_generator_in_background(
        self, tool: Tool, data: FunctionCallItem, arguments: dict[str, Any]
    ) -> None:
        task = asyncio.create_task(
            self._stream_async_generator_results(tool, data, arguments)
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _stream_async_generator_results(
        self, tool: Tool, data: FunctionCallItem, arguments: dict[str, Any]
    ) -> None:
        self.logger.info("Starting async generator tool: %s", tool.name)

        async_generator = await tool.execute(arguments)

        async for chunk in async_generator:
            await self._message_manager.send_execution_message(chunk)
            self.logger.debug("Generator yielded chunk: %s", chunk)

        self.logger.info("Async generator tool '%s' completed", tool.name)

    async def _send_result_and_notify(
        self,
        call_data: FunctionCallItem,
        result: Any,
        response_instruction: str | None = None,
    ) -> None:
        function_call_result = FunctionCallResult(
            tool_name=call_data.name,
            call_id=call_data.call_id,
            output=result,
            response_instruction=response_instruction,
        )
        await self._message_manager.send_tool_result(function_call_result)
        self.logger.info("Tool result sent for: %s", call_data.name)

        await self._notify_tool_finished()

    async def _notify_tool_finished(self) -> None:
        self.logger.info("Tool finished - notifying system")
        await self._event_bus.publish_async(
            VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT
        )

    def _build_final_arguments(
        self, func: callable, llm_arguments: dict[str, Any]
    ) -> dict[str, Any]:
        signature = inspect.signature(func)
        final_arguments = llm_arguments.copy()
        special_param_names = set(SpecialToolParameters.model_fields.keys())

        for param_name, param in signature.parameters.items():
            if self._should_skip_parameter(param_name, final_arguments):
                continue

            if param_name in special_param_names:
                self._inject_special_parameter_if_available(
                    param_name, param, final_arguments
                )

        return final_arguments

    def _should_skip_parameter(
        self, param_name: str, current_arguments: dict[str, Any]
    ) -> bool:
        return param_name in current_arguments or param_name in ("self", "cls")

    def _inject_special_parameter_if_available(
        self,
        param_name: str,
        param: inspect.Parameter,
        arguments: dict[str, Any],
    ) -> None:
        injected_value = getattr(self._special_tool_parameters, param_name, None)

        if injected_value is not None:
            arguments[param_name] = injected_value
            self.logger.debug("Injected special parameter: %s", param_name)
        elif param.default == inspect.Parameter.empty:
            raise ValueError(
                f"Required special parameter '{param_name}' is not available"
            )
