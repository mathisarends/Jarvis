from __future__ import annotations

import asyncio
import inspect
from typing import Any

from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.tool import Tool
from agent.realtime.tools.views import (
    FunctionCallItem,
    FunctionCallResult,
    SpecialToolParameters,
)

from agent.events import EventBus
from agent.realtime.messaging.message_manager import RealtimeMessageManager
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


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
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._event_bus = event_bus

        self._event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_STARTED_TOOL_CALL, self._handle_tool_call
        )

    async def _handle_tool_call(self, data: FunctionCallItem) -> None:
        function_name = data.name
        llm_arguments = data.arguments or {}

        self.logger.debug("Executing tool: %s", function_name)

        tool = self._tool_registry.get(function_name)

        if tool.execution_message:
            await self._message_manager.send_execution_message(tool.execution_message)

        await self._execute_tool(tool, data, llm_arguments)

    async def _execute_tool(
        self, tool: Tool, data: FunctionCallItem, llm_arguments: dict[str, Any]
    ) -> None:
        final_arguments = self._inject_special_parameters(tool.function, llm_arguments)

        if tool.is_async_generator:
            task = asyncio.create_task(
                self._execute_async_generator(tool, data, final_arguments)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        else:
            result = await tool.execute(final_arguments)
            await self._send_tool_result(data, result, tool.response_instruction)

    async def _send_tool_result(
        self,
        data: FunctionCallItem,
        result: Any,
        response_instruction: str | None = None,
    ) -> None:
        function_call_result = FunctionCallResult(
            tool_name=data.name,
            call_id=data.call_id,
            output=result,
            response_instruction=response_instruction,
        )
        await self._message_manager.send_tool_result(function_call_result)
        await self._publish_tool_call_result_event()

    async def _publish_tool_call_result_event(self) -> None:
        await self._event_bus.publish_async(
            VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT
        )

    async def _execute_async_generator(
        self, tool: Tool, data: FunctionCallItem, final_arguments: dict[str, Any]
    ) -> None:
        async_generator = await tool.execute(final_arguments)

        async for chunk in async_generator:
            await self._message_manager.send_execution_message(chunk)

    async def _handle_tool_error(
        self, data: FunctionCallItem, error: Exception, context: str
    ) -> None:
        self.logger.error("Error %s: %s", context, error, exc_info=True)

        function_call_result = FunctionCallResult(
            tool_name=data.name,
            call_id=data.call_id,
            output=f"Error: {error!s}",
            response_instruction="This is an error message that should be communicated to the user",
        )
        await self._message_manager.send_tool_result(function_call_result)
        await self._publish_tool_call_result_event()

    def _inject_special_parameters(
        self, func: callable, llm_arguments: dict[str, Any]
    ) -> dict[str, Any]:
        signature = inspect.signature(func)
        final_arguments = llm_arguments.copy()
        special_param_names = set(SpecialToolParameters.model_fields.keys())

        for param_name, param in signature.parameters.items():
            if param_name in final_arguments or param_name in ("self", "cls"):
                continue

            if param_name in special_param_names:
                injected_value = getattr(
                    self._special_tool_parameters, param_name, None
                )

                if injected_value is not None:
                    final_arguments[param_name] = injected_value
                elif param.default == inspect.Parameter.empty:
                    raise ValueError(
                        f"Required special parameter '{param_name}' is not available"
                    )

        return final_arguments
