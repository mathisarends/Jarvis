from __future__ import annotations

from agents import FunctionTool
from agent.realtime.views import FunctionTool as PydanticTool

from shared.logging_mixin import LoggingMixin
from shared.singleton_decorator import singleton


@singleton
class ToolRegistry(LoggingMixin):
    """
    Registry for OpenAI Agents SDK FunctionTool objects.
    - Stores FunctionTool instances
    - Exposes OpenAI Realtime-compatible tool schemas (for session.update)
    - Supports optional "early message" per tool
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
        """
        Register a FunctionTool.

        Args:
            tool: FunctionTool instance
            return_early_message: optional early-response text
        """
        self._validate_tool(tool)
        
        if tool.name in self._tools:
            raise ValueError(f"A tool with the name '{tool.name}' is already registered.")

        self._tools[tool.name] = tool
        if return_early_message:
            self._early_messages[tool.name] = return_early_message

        self.logger.info("Tool '%s' successfully registered.", tool.name)

    def unregister_tool(self, tool_name: str) -> bool:
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._early_messages.pop(tool_name, None)
            self.logger.info("Tool '%s' removed from registry.", tool_name)
            return True
        self.logger.warning("Tool '%s' could not be removed (not found).", tool_name)
        return False

    def get_tool(self, tool_name: str) -> FunctionTool | None:
        return self._tools.get(tool_name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_tools(self) -> list[FunctionTool]:
        return list(self._tools.values())

    def get_early_message(self, tool_name: str) -> str | None:
        return self._early_messages.get(tool_name)

    # ---------- Schema export for Realtime session.update ----------

    def get_openai_schema(self) -> list[PydanticTool]:
        """
        Build OpenAI Realtime-compatible tool descriptors as Pydantic models.
        Returns a list of FunctionTool objects for structured tool configuration.
        """
        tools: list[PydanticTool] = []
        for ft in self._tools.values():
            params_schema = getattr(ft, "params_json_schema", None)
            if not params_schema:
                # fallback: empty object schema
                params_schema = {"type": "object", "properties": {}, "required": []}

            # enforce additionalProperties=False for stricter validation
            if isinstance(params_schema, dict) and "additionalProperties" not in params_schema:
                params_schema = {**params_schema, "additionalProperties": False}

            # Create Pydantic FunctionTool model
            function_tool = PydanticTool(
                type="function",
                name=ft.name,
                description=getattr(ft, "description", None) or "No description provided.",
                parameters=params_schema,
            )
            tools.append(function_tool)
        
        return tools

    def _validate_tool(self, tool: FunctionTool) -> None:
        """
        Validate that the tool has required attributes.
        """
        if not hasattr(tool, 'name') or not tool.name:
            raise ValueError("Tool must have a valid name")
        
        if not hasattr(tool, 'description') or not tool.description:
            raise ValueError("Tool must have a description")
        
        if not hasattr(tool, 'params_json_schema'):
            raise ValueError("Tool must have params_json_schema")