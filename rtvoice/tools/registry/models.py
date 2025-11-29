import inspect
from collections.abc import Callable
from typing import Any

from rtvoice.tools.models import FunctionParameters, FunctionTool


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        schema: FunctionParameters,
        response_instruction: str | None = None,
        execution_message: str | None = None,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema
        self.response_instruction = response_instruction
        self.execution_message = execution_message

    async def execute(self, arguments: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(self.function):
            return await self.function(**arguments)
        else:
            return self.function(**arguments)

    def to_pydantic(self) -> FunctionTool:
        return FunctionTool(
            name=self.name,
            description=self.description,
            parameters=self.schema,
        )
