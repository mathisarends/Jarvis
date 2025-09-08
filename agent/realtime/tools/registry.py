from typing import Optional, Callable

from agent.realtime.events.client.session_update import FunctionTool
from agent.realtime.tools.tool import Tool

from shared.logging_mixin import LoggingMixin

# bei mir ist dsa quasi das hier den wrapper den ich hier für tool geschireben habe ich will aber dass das hier jetzt in der registry hier stattfindet weißt du:
class ToolRegistry(LoggingMixin):
    """
    Registry for OpenAI Agents SDK FunctionTool objects.
    """

    """Simple tool registry."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_openai_schema(self) -> list[FunctionTool]:
        """Convert all tools to Pydantic FunctionTool format."""
        return [tool.to_pydantic() for tool in self._tools.values()]