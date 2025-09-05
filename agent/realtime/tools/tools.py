from datetime import datetime
from agent.realtime.tools.tool import tool

from agent.realtime.tools.weather import get_weather_for_current_location


@tool(description="Get the current local time")
def get_current_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


@tool(
    description="Get comprehensive 3-day weather forecast for your current location. Automatically detects your location via IP and fetches detailed weather data including current conditions and hourly forecasts.",
    loading_message="Featching weather data...",
    result_context="Focus on today's weather progression and upcoming changes. Leave out the rest of the forecast.",
    long_running=False,
)
async def get_weather() -> str:
    """Get weather report for current location with 3-day forecast."""
    return await get_weather_for_current_location()
