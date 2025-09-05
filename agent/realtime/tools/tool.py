from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, get_type_hints, get_origin, get_args
import collections.abc

from agent.realtime.events.client.session_update import FunctionTool
from shared.logging_mixin import LoggingMixin


class Tool(LoggingMixin):
    """Simple tool wrapper with loading message and result context support."""

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        schema: dict[str, Any],
        result_context: Optional[str] = None,
        is_generator: bool = False,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema
        self.result_context = result_context
        self.is_generator = is_generator

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


def tool(
    description: str,
    name: Optional[str] = None,
    result_context: Optional[str] = None,
):
    """
    Simple tool decorator that extracts parameters from function signature.

    Args:
        description: Required description of the tool's functionality
        name: Optional custom name for the tool (defaults to function name)
        result_context: Context information for handling the tool result (e.g. "Das Ergebnis sollte als Markdown formatiert werden")

    Usage:
    @tool("Get the current local time")
    def get_current_time() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @tool(
        "Search for information online",
        result_context="Die gefundenen Informationen sollten kritisch bewertet und zusammengefasst werden."
    )
    def web_search(query: str) -> str:
        # Search implementation
        pass

    @tool(
        "Analyze data from uploaded file",
        e ="Die Analyse sollte in strukturierter Form mit Diagrammen präsentiert werden."
    )
    def analyze_file(file_path: str) -> dict:
        # Analysis implementation
        pass
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        schema = _generate_schema_from_function(func)
        is_generator = _is_generator_function(func)

        return Tool(
            name=tool_name,
            description=description,
            function=func,
            schema=schema,
            result_context=result_context,
            is_generator=is_generator,
        )

    return decorator


def _is_generator_function(func: Callable) -> bool:
    """Check if a function is a generator based on its return type hints."""
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return")
    return _is_generator_type(return_type)


def _is_generator_type(return_type: Any) -> bool:
    """Check if the return type indicates a generator function."""
    if return_type is None:
        return False

    # Handle typing module types
    origin = get_origin(return_type)
    if origin is not None:
        # Check for Generator or AsyncGenerator
        if origin in (collections.abc.Generator, collections.abc.AsyncGenerator):
            return True
        # Check for typing.Generator or typing.AsyncGenerator
        try:
            import typing

            if origin in (typing.Generator, typing.AsyncGenerator):
                return True
        except ImportError:
            pass

    # Check for direct Generator/AsyncGenerator types
    try:
        import types

        if hasattr(types, "GeneratorType") and return_type == types.GeneratorType:
            return True
        if (
            hasattr(types, "AsyncGeneratorType")
            and return_type == types.AsyncGeneratorType
        ):
            return True
    except ImportError:
        pass

    return False


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
