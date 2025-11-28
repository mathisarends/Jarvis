import collections.abc
import inspect
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from agent.events.schemas import (
    FunctionParameterProperty,
    FunctionParameters,
    FunctionTool,
    MCPTool,
)
from agent.tools.models import SpecialToolParameters
from agent.tools.registry.models import Tool


class ToolRegistry:
    def __init__(
        self,
        mcp_tools: list[MCPTool] | None = None,
    ):
        self.mcp_tools = mcp_tools or []
        self._tools: dict[str, Tool] = {}

    def action(
        self,
        description: str,
        name: str | None = None,
        response_instruction: str | None = None,
        execution_message: str | None = None,
    ):
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            schema = self._generate_schema_from_function(func)

            if hasattr(self, func.__name__):
                bound_func = getattr(self, func.__name__)
            else:
                bound_func = func

            tool = Tool(
                name=tool_name,
                description=description,
                function=bound_func,
                schema=schema,
                response_instruction=response_instruction,
                execution_message=execution_message,
            )

            self._register(tool)
            return func

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_openai_schema(self) -> list[FunctionTool]:
        return [tool.to_pydantic() for tool in self._tools.values()] + self.mcp_tools

    def _register(self, tool: Tool):
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def _generate_schema_from_function(self, func: Callable) -> FunctionParameters:
        signature = inspect.signature(func)
        type_hints = get_type_hints(
            func, include_extras=True
        )  # include_extras=True for Annotated

        properties: dict[str, FunctionParameterProperty] = {}
        required_params: list[str] = []

        special_param_names = set(SpecialToolParameters.model_fields.keys())
        special_param_types = self._extract_special_parameter_types()

        for param_name, param in signature.parameters.items():
            # Skip 'self'/'cls'
            if param_name in ("self", "cls"):
                continue

            # EXCLUDE: injected special parameters
            if param_name in special_param_names:
                continue

            param_annotation = type_hints.get(param_name, str)

            # EXCLUDE: type-based filtering (more robust)
            if self._is_special_type(param_annotation, special_param_types):
                continue

            # Extract actual type and description from potentially Annotated type
            actual_type, description = self._extract_type_and_description(
                param_annotation
            )

            # Include in schema
            properties[param_name] = self._type_to_json_schema(actual_type, description)

            if param.default == inspect.Parameter.empty:
                required_params.append(param_name)

        return FunctionParameters(
            type="object",
            strict=True,
            properties=properties,
            required=required_params,
        )

    def _extract_special_parameter_types(self) -> set[type]:
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

    def _extract_type_and_description(self, type_hint: Any) -> tuple[Any, str | None]:
        """
        Extract actual type and description from potentially Annotated type.
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

    def _is_special_type(self, type_hint: Any, special_param_types: set[type]) -> bool:
        # Extract actual type from Annotated if needed
        actual_type, _ = self._extract_type_and_description(type_hint)

        # Optional/Union entpacken
        origin = get_origin(actual_type)
        if origin is Union:
            union_args = get_args(actual_type)
            non_none_args = [arg for arg in union_args if arg is not type(None)]
            return any(
                self._is_special_type(arg, special_param_types) for arg in non_none_args
            )

        # Direkte Typzuordnung
        return isinstance(actual_type, type) and actual_type in special_param_types

    def _type_to_json_schema(
        self, python_type: Any, description: str | None = None
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
                return self._type_to_json_schema(union_args[0], description)
            # Complex unions: compact fallback
            return FunctionParameterProperty(type="string", description=description)

        if origin is list:
            return FunctionParameterProperty(type="array", description=description)

        # Dict[K, V]
        if origin is dict:
            return FunctionParameterProperty(type="object", description=description)

        if origin in (
            collections.abc.Sequence,
            collections.abc.Iterable,
            collections.abc.Collection,
        ):
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
