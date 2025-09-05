from datetime import datetime
from agent.realtime.tools.tool import tool


@tool(description="Get the current local time")
def get_current_time() -> str:
    return datetime.now().strftime("%H:%M:%S")