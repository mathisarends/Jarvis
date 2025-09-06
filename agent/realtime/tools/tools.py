from datetime import datetime
from agent.realtime.tools.tool import tool

from agent.realtime.tools.weather import get_weather_for_current_location
from agent.realtime.tools.web_search import run_web_search_agent
from audio.player.audio_manager import AudioManager


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
    description=(
        "Delegates a task to a specialized web search agent that automatically optimizes the query with contextual information and returns aggregated search results from the web."
    ),
)
async def delegate_task_to_web_search_agent(query: str) -> str:
    """Perform a web search and return the aggregated results."""
    return await run_web_search_agent(query)


@tool("Play a sound")
async def play_sound(name: str, audio_manager: AudioManager) -> str:
    audio_manager.get_strategy().play_startup_sound()
