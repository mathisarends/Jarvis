from __future__ import annotations
from typing import Optional, Any

from agent.realtime.tools.simple_tool import SimpleTool

from agent.realtime.views import FunctionTool
from shared.logging_mixin import LoggingMixin
from shared.singleton_decorator import singleton


@singleton
class ToolRegistry(LoggingMixin):
    """
    Registry for OpenAI Agents SDK FunctionTool objects.
    """

    """Simple tool registry."""

    def __init__(self):
        self._tools: dict[str, SimpleTool] = {}

    def register(self, tool: SimpleTool):
        """Register a tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[SimpleTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        """List all tool names."""
        return list(self._tools.keys())

    def get_all(self) -> list[SimpleTool]:
        """Get all tools."""
        return list(self._tools.values())

    def get_openai_schema(self) -> list[FunctionTool]:
        """Convert all tools to Pydantic FunctionTool format."""
        return [tool.to_pydantic() for tool in self._tools.values()]

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return await tool.execute(arguments)
