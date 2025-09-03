from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field
from agents import function_tool


class TimeQuery(BaseModel):
    format: Literal["time", "date", "datetime"] = Field(
        "time", description="Which format to return: 'time', 'date', or 'datetime'."
    )


@function_tool
def get_current_time(query: TimeQuery) -> dict:
    """
    Get the current local time/date.
    """
    now = datetime.now()

    if query.format == "date":
        value = now.strftime("%Y-%m-%d")
    elif query.format == "datetime":
        value = now.isoformat(sep=" ", timespec="seconds")
    else:  # default: time
        value = now.strftime("%H:%M:%S")

    return {"format": query.format, "value": value}