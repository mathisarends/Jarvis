from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, get_type_hints, get_origin, get_args, Union
import collections.abc

from agent.realtime.events.client.session_update import FunctionTool
from agent.realtime.tools.views import SpecialToolParameters
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
        result_context: Context information for handling the tool result
                        (e.g. "Das Ergebnis sollte als Markdown formatiert werden")

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
        result_context="Die Analyse sollte in strukturierter Form mit Diagrammen präsentiert werden."
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

    origin = get_origin(return_type)
    if origin is not None:
        # collections.abc.Generator / AsyncGenerator
        if origin in (collections.abc.Generator, collections.abc.AsyncGenerator):
            return True
        # typing.Generator / typing.AsyncGenerator (kompatibel)
        try:
            import typing

            if origin in (typing.Generator, typing.AsyncGenerator):
                return True
        except ImportError:
            pass

    # Fallback auf types.* (selten relevant)
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


def _extract_special_parameter_types() -> set[type]:
    """Extract types from SpecialToolParameters for filtering."""
    special_types: set[type] = set()

    for field in SpecialToolParameters.model_fields.values():
        annotation = field.annotation
        # Resolve Optional[T] / Union[T, None]
        origin = get_origin(annotation)
        if origin is Union:
            non_none_types = [
                arg for arg in get_args(annotation) if arg is not type(None)
            ]
            if non_none_types:
                annotation = non_none_types[0]

        if isinstance(annotation, type):
            special_types.add(annotation)

    return special_types


def _is_special_type(type_hint: Any, special_param_types: set[type]) -> bool:
    """Check if a type hint represents a special parameter type."""
    # Optional/Union entpacken
    origin = get_origin(type_hint)
    if origin is Union:
        union_args = get_args(type_hint)
        non_none_args = [arg for arg in union_args if arg is not type(None)]
        return any(_is_special_type(arg, special_param_types) for arg in non_none_args)

    # Direkte Typzuordnung
    return isinstance(type_hint, type) and type_hint in special_param_types


def _generate_schema_from_function(func: Callable) -> dict[str, Any]:
    """
    Generate JSON schema from function signature for LLM tools.

    Important:
    - Filters out all parameters defined in SpecialToolParameters
      (both by name and type) so they don't reach the LLM schema.
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required_params: list[str] = []

    # Extract special parameter names and types for filtering
    special_param_names = set(SpecialToolParameters.model_fields.keys())
    special_param_types = _extract_special_parameter_types()

    for param_name, param in signature.parameters.items():
        # Skip 'self'/'cls'
        if param_name in ("self", "cls"):
            continue

        # EXCLUDE: name-based filtering
        if param_name in special_param_names:
            continue

        param_annotation = type_hints.get(param_name, str)

        # EXCLUDE: type-based filtering (more robust)
        if _is_special_type(param_annotation, special_param_types):
            continue

        # Include in schema
        properties[param_name] = _type_to_json_schema(param_annotation)

        if param.default == inspect.Parameter.empty:
            required_params.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required_params,
        "additionalProperties": False,
    }


def _type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """Convert Python type to JSON schema."""
    origin = get_origin(python_type)

    # Optional[T] / Union[T, None]
    if origin is Union:
        union_args = [arg for arg in get_args(python_type) if arg is not type(None)]
        if len(union_args) == 1:
            # Optional[T] -> T
            return _type_to_json_schema(union_args[0])
        # Complex unions: compact fallback (alternatively: build anyOf)
        return {"type": "string"}

    # List[T]
    if origin is list:
        list_args = get_args(python_type)
        item_type = list_args[0] if list_args else str
        return {"type": "array", "items": _type_to_json_schema(item_type)}

    # Dict[K, V]
    if origin is dict:
        return {"type": "object", "additionalProperties": True}

    # collections.abc.Sequence[T] / Iterable[T] etc. – optional if needed:
    if origin in (
        collections.abc.Sequence,
        collections.abc.Iterable,
        collections.abc.Collection,
    ):
        collection_args = get_args(python_type)
        item_type = collection_args[0] if collection_args else str
        return {"type": "array", "items": _type_to_json_schema(item_type)}

    # Primitive types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object", "additionalProperties": True},
    }

    mapped_type = type_mapping.get(python_type)
    if mapped_type:
        return mapped_type

    # Unknown / complex types (including Pydantic models) → generic as object
    try:
        from pydantic import BaseModel

        if isinstance(python_type, type) and issubclass(python_type, BaseModel):
            return {"type": "object", "additionalProperties": True}
    except Exception:
        pass

    return {"type": "string"}
