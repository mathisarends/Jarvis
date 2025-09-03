from __future__ import annotations

from datetime import datetime
from agents import function_tool


@function_tool
def get_current_time() -> str:
    """
    Get the current local time.
    """
    now = datetime.now()
    return now.strftime("%H:%M:%S")