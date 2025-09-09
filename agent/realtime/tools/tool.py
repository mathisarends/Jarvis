from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Optional,
)

from agent.realtime.events.client.session_update import (
    FunctionTool,
    FunctionParameters,
)
from shared.logging_mixin import LoggingMixin


class Tool(LoggingMixin):
    """Simple tool wrapper with loading message and result context support."""

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        schema: FunctionParameters,
        response_instruction: Optional[str] = None,
        execution_message: Optional[str] = None,
        is_async_generator: bool = False,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema
        self.response_instruction = response_instruction
        self.execution_message = execution_message
        self.is_async_generator = is_async_generator

    async def execute(self, arguments: dict[str, Any]) -> Any:
        """Execute the tool function."""
        if inspect.iscoroutinefunction(self.function):
            return await self.function(**arguments)
        else:
            return self.function(**arguments)

    def to_pydantic(self) -> FunctionTool:
        """Convert to Pydantic FunctionTool."""
        return FunctionTool(
            type="function",
            name=self.name,
            description=self.description,
            parameters=self.schema,
        )
