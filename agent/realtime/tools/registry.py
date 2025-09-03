from __future__ import annotations

from agents import FunctionTool
from agent.realtime.views import FunctionTool as PydanticTool

from shared.logging_mixin import LoggingMixin
from shared.singleton_decorator import singleton


@singleton
class ToolRegistry(LoggingMixin):
    """
    Registry for OpenAI Agents SDK FunctionTool objects.
    """

    def __init__(self) -> None:
        self._tools: dict[str, FunctionTool] = {}
        self._early_messages: dict[str, str] = {}

    def register_tool(
        self,
        tool: FunctionTool,
        *,
        return_early_message: str = "",
    ) -> None:
        """Register a FunctionTool."""
        self._validate_tool(tool)

        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")

        self._tools[tool.name] = tool
        if return_early_message:
            self._early_messages[tool.name] = return_early_message

        self.logger.info("Tool '%s' registered.", tool.name)

    def unregister_tool(self, tool_name: str) -> bool:
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._early_messages.pop(tool_name, None)
            self.logger.info("Tool '%s' removed.", tool_name)
            return True
        return False

    def get_tool(self, tool_name: str) -> FunctionTool | None:
        return self._tools.get(tool_name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_tools(self) -> list[FunctionTool]:
        return list(self._tools.values())

    def get_early_message(self, tool_name: str) -> str | None:
        return self._early_messages.get(tool_name)

    def get_openai_schema(self) -> list[PydanticTool]:
        """Build OpenAI Realtime-compatible tool schemas."""
        tools: list[PydanticTool] = []
        
        for ft in self._tools.values():
            # Get original schema
            schema = getattr(ft, "params_json_schema", None)
            if not schema:
                schema = {"type": "object", "properties": {}, "required": []}

            # Clean schema for OpenAI
            cleaned_schema = self._clean_for_openai(schema)
            
            # Create tool
            function_tool = PydanticTool(
                type="function",
                name=ft.name,
                description=getattr(ft, "description", None) or "No description provided.",
                parameters=cleaned_schema,
            )
            tools.append(function_tool)

        return tools

    def _clean_for_openai(self, schema: dict) -> dict:
        """Clean schema for OpenAI Realtime API."""
        if not isinstance(schema, dict):
            return schema

        # Remove problematic Pydantic fields
        cleaned = self._remove_unwanted_fields(schema)
        
        # Add required OpenAI fields
        if "type" not in cleaned:
            cleaned["type"] = "object"
        cleaned["strict"] = True
        cleaned["additionalProperties"] = False
        
        return cleaned

    def _remove_unwanted_fields(self, obj):
        """Remove fields that OpenAI doesn't like."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                # Skip these Pydantic fields
                if key in ["title", "default", "$defs", "allOf", "anyOf"]:
                    continue
                result[key] = self._remove_unwanted_fields(value)
            return result
        elif isinstance(obj, list):
            return [self._remove_unwanted_fields(item) for item in obj]
        else:
            return obj

    def _validate_tool(self, tool: FunctionTool) -> None:
        """Validate tool has required attributes."""
        if not hasattr(tool, "name") or not tool.name:
            raise ValueError("Tool must have a valid name")
        if not hasattr(tool, "description") or not tool.description:
            raise ValueError("Tool must have a description")
        if not hasattr(tool, "params_json_schema"):
            raise ValueError("Tool must have params_json_schema")