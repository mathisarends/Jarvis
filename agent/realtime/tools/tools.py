from datetime import datetime
from agent.realtime.tools.simple_tool import tool


@tool()
def get_current_time() -> str:
    """Get the current local time."""
    return datetime.now().strftime("%H:%M:%S")
