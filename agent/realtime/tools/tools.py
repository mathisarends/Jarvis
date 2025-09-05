from datetime import datetime
from typing import AsyncGenerator
from agent.realtime.tools.tool import tool

from agent.realtime.tools.weather import get_weather_for_current_location
from agent.realtime.tools.browser_search import perform_browser_search


@tool(description="Get the current local time")
def get_current_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


@tool(
    description="Get comprehensive 3-day weather forecast for your current location. Automatically detects your location via IP and fetches detailed weather data including current conditions and hourly forecasts.",
    result_context="Focus on today's weather progression and upcoming changes. Leave out the rest of the forecast.",
)
async def get_weather() -> str:
    """Get weather report for current location with 3-day forecast."""
    return await get_weather_for_current_location()


@tool(
    description="Performs web search using an automated browser"
)
async def perform_browser_search_tool(query: str) -> AsyncGenerator[str, None]:
    """Perform a browser search and yield results in real-time."""
    async for message in perform_browser_search(query):
        yield message