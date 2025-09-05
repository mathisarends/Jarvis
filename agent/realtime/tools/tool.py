
from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, get_type_hints, get_origin, get_args

from agent.realtime.views import FunctionTool
from shared.logging_mixin import LoggingMixin


class Tool(LoggingMixin):
    """Simple tool wrapper with loading message and result context support."""

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        schema: dict[str, Any],
        long_running: bool = False,
        loading_message: Optional[str] = None,
        result_context: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema
        self.long_running = long_running
        self.loading_message = loading_message
        self.result_context = result_context

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

    def get_loading_message(self) -> Optional[str]:
        """Get the loading message for this tool."""
        return self.loading_message

    def get_result_context(self) -> Optional[str]:
        """Get the result context for this tool."""
        return self.result_context


def tool(
    description: str, 
    name: Optional[str] = None, 
    long_running: bool = False,
    loading_message: Optional[str] = None,
    result_context: Optional[str] = None,
):
    """
    Simple tool decorator that extracts parameters from function signature.

    Args:
        description: Required description of the tool's functionality
        name: Optional custom name for the tool (defaults to function name)
        long_running: Flag indicating if this is a long-running operation
        loading_message: Message to show while the tool is running (e.g. "Ich schaue das eben im Browser nach...")
        result_context: Context information for handling the tool result (e.g. "Das Ergebnis sollte als Markdown formatiert werden")

    Usage:
    @tool("Get the current local time")
    def get_current_time() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @tool(
        "Search for information online", 
        long_running=True,
        loading_message="Ich durchsuche gerade das Internet nach Informationen...",
        result_context="Die gefundenen Informationen sollten kritisch bewertet und zusammengefasst werden."
    )
    def web_search(query: str) -> str:
        # Search implementation
        pass

    @tool(
        "Analyze data from uploaded file",
        loading_message="Ich analysiere die hochgeladene Datei...",
        result_context="Die Analyse sollte in strukturierter Form mit Diagrammen präsentiert werden."
    )
    def analyze_file(file_path: str) -> dict:
        # Analysis implementation
        pass
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        schema = _generate_schema_from_function(func)

        default_loading_message = None
        if long_running and loading_message is None:
            default_loading_message = f"Executing {tool_name} ..."

        return Tool(
            name=tool_name,
            description=description,
            function=func,
            schema=schema,
            long_running=long_running,
            loading_message=loading_message or default_loading_message,
            result_context=result_context,
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