from __future__ import annotations

import json
import asyncio
import inspect
from typing import Any, Callable, Optional, get_type_hints, get_origin, get_args
from datetime import datetime

from agent.realtime.views import FunctionTool


class SimpleTool:
    """Simple tool wrapper."""

    def __init__(
        self, name: str, description: str, function: Callable, schema: dict[str, Any]
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema

    async def execute(self, arguments: dict[str, Any]) -> Any:
        """Execute the tool function."""
        try:
            if inspect.iscoroutinefunction(self.function):
                return await self.function(**arguments)
            else:
                return self.function(**arguments)
        except Exception as e:
            raise Exception(f"Tool '{self.name}' execution failed: {str(e)}")

    def to_pydantic(self) -> FunctionTool:
        """Convert to Pydantic FunctionTool."""
        return FunctionTool(
            type="function",
            name=self.name,
            description=self.description,
            parameters=self.schema,
        )


def tool(name: Optional[str] = None):
    """
    Simple tool decorator that extracts parameters from function signature
    and description from docstring.

    Usage:
    @tool()
    def get_current_time() -> str:
        '''Get the current local time.'''
        return datetime.now().strftime("%H:%M:%S")

    @tool("custom_name")
    def my_function(text: str, count: int = 1) -> str:
        '''Repeat text multiple times.'''
        return text * count
    """

    def decorator(func: Callable) -> SimpleTool:
        tool_name = name or func.__name__
        tool_description = (func.__doc__ or "").strip() or "No description provided"
        schema = _generate_schema_from_function(func)

        return SimpleTool(
            name=tool_name, description=tool_description, function=func, schema=schema
        )

    return decorator


def _generate_schema_from_function(func: Callable) -> dict[str, Any]:
    """Generate JSON schema from function signature."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in signature.parameters.items():
        # Skip self and cls parameters
        if param_name in ("self", "cls"):
            continue

        param_type = type_hints.get(param_name, str)
        param_schema = _type_to_json_schema(param_type)

        properties[param_name] = param_schema

        # Add to required if no default value
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert Python type to JSON schema."""
    origin = get_origin(python_type)

    # Handle Optional types (Union[T, None])
    if origin is type(int | None):  # Union type
        args = get_args(python_type)
        if len(args) == 2 and type(None) in args:
            # This is Optional[T], get the non-None type
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return _type_to_json_schema(non_none_type)

    # Handle List types
    if origin is list:
        args = get_args(python_type)
        item_type = args[0] if args else str
        return {"type": "array", "items": _type_to_json_schema(item_type)}

    # Handle Dict types
    if origin is dict:
        return {"type": "object", "additionalProperties": True}

    # Basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object", "additionalProperties": True},
    }

    return type_mapping.get(python_type, {"type": "string"})
