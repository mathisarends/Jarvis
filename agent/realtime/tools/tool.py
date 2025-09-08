from __future__ import annotations

import inspect
from typing import (
    Any,
    Callable,
    Optional,
    get_type_hints,
    get_origin,
    get_args,
    Union,
    Annotated,
)
import collections.abc

from agent.realtime.events.client.session_update import (
    FunctionTool,
    FunctionParameters,
    FunctionParameterProperty,
)
from agent.realtime.tools.views import SpecialToolParameters
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
        is_generator: bool = False,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.schema = schema
        self.response_instruction = response_instruction
        self.execution_message = execution_message
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
    response_instruction: Optional[str] = None,
    execution_message: Optional[str] = None,
):
    """
    Simple tool decorator that extracts parameters from function signature.

    Args:
        description: Required description of the tool's functionality
        name: Optional custom name for the tool (defaults to function name)
        response_instruction: Context information for handling the tool result
                        (e.g. "Das Ergebnis sollte als Markdown formatiert werden")

    Usage:
    @tool("Get the current local time")
    def get_current_time() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @tool(
        "Search for information online",
        response_instruction="Die gefundenen Informationen sollten kritisch bewertet und zusammengefasst werden."
    )
    def web_search(query: str) -> str:
        # Search implementation
        pass

    # With parameter descriptions using Annotated:
    @tool("Search with detailed parameters")
    def search_detailed(
        query: Annotated[str, "The search query to execute"],
        limit: Annotated[int, "Maximum number of results to return"] = 10
    ) -> str:
        # Search implementation
        pass

    @tool(
        "Analyze data from uploaded file",
        response_instruction="Die Analyse sollte in strukturierter Form mit Diagrammen präsentiert werden."
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
            response_instruction=response_instruction,
            execution_message=execution_message,
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


def _extract_type_and_description(type_hint: Any) -> tuple[Any, str | None]:
    """
    Extract actual type and description from potentially Annotated type.

    Args:
        type_hint: Type hint that might be Annotated[T, "description"]

    Returns:
        Tuple of (actual_type, description). Description is None if not annotated.

    Examples:
        Annotated[str, "A query string"] -> (str, "A query string")
        str -> (str, None)
        Annotated[int, "Count", "more metadata"] -> (int, "Count")
    """
    if get_origin(type_hint) is Annotated:
        args = get_args(type_hint)
        actual_type = args[0]
        # Look for first string argument as description
        description = None
        for arg in args[1:]:
            if isinstance(arg, str):
                description = arg
                break
        return actual_type, description

    return type_hint, None


def _is_special_type(type_hint: Any, special_param_types: set[type]) -> bool:
    """Check if a type hint represents a special parameter type."""
    # Extract actual type from Annotated if needed
    actual_type, _ = _extract_type_and_description(type_hint)

    # Optional/Union entpacken
    origin = get_origin(actual_type)
    if origin is Union:
        union_args = get_args(actual_type)
        non_none_args = [arg for arg in union_args if arg is not type(None)]
        return any(_is_special_type(arg, special_param_types) for arg in non_none_args)

    # Direkte Typzuordnung
    return isinstance(actual_type, type) and actual_type in special_param_types


def _generate_schema_from_function(func: Callable) -> FunctionParameters:
    """
    Generate JSON schema from function signature for LLM tools.

    Important:
    - Filters out all parameters defined in SpecialToolParameters
      (both by name and type) so they don't reach the LLM schema.
    - Supports Annotated[Type, "description"] for parameter descriptions.
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(
        func, include_extras=True
    )  # include_extras=True for Annotated

    properties: dict[str, FunctionParameterProperty] = {}
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

        # Extract actual type and description from potentially Annotated type
        actual_type, description = _extract_type_and_description(param_annotation)

        # Include in schema
        properties[param_name] = _type_to_json_schema(actual_type, description)

        if param.default == inspect.Parameter.empty:
            required_params.append(param_name)

    return FunctionParameters(
        type="object",
        strict=True,
        properties=properties,
        required=required_params,
    )


def _type_to_json_schema(
    python_type: Any, description: str | None = None
) -> FunctionParameterProperty:
    """
    Convert Python type to JSON schema property.

    Args:
        python_type: The Python type to convert
        description: Optional description for the parameter
    """
    origin = get_origin(python_type)

    # Optional[T] / Union[T, None]
    if origin is Union:
        union_args = [arg for arg in get_args(python_type) if arg is not type(None)]
        if len(union_args) == 1:
            # Optional[T] -> T (keep description)
            return _type_to_json_schema(union_args[0], description)
        # Complex unions: compact fallback
        return FunctionParameterProperty(type="string", description=description)

    # List[T]
    if origin is list:
        list_args = get_args(python_type)
        item_type = list_args[0] if list_args else str
        # Note: For now we don't support nested items in FunctionParameterProperty
        # If needed later, we could add an items field to the model
        return FunctionParameterProperty(type="array", description=description)

    # Dict[K, V]
    if origin is dict:
        return FunctionParameterProperty(type="object", description=description)

    # collections.abc.Sequence[T] / Iterable[T] etc. – optional if needed:
    if origin in (
        collections.abc.Sequence,
        collections.abc.Iterable,
        collections.abc.Collection,
    ):
        collection_args = get_args(python_type)
        item_type = collection_args[0] if collection_args else str
        return FunctionParameterProperty(type="array", description=description)

    # Primitive types
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    json_type = type_mapping.get(python_type)
    if json_type:
        return FunctionParameterProperty(type=json_type, description=description)

    # Unknown / complex types (including Pydantic models) → generic as object
    try:
        from pydantic import BaseModel

        if isinstance(python_type, type) and issubclass(python_type, BaseModel):
            return FunctionParameterProperty(type="object", description=description)
    except Exception:
        pass

    return FunctionParameterProperty(type="string", description=description)
