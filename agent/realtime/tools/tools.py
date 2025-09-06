from datetime import datetime
from typing import Annotated
from agent.realtime.tools.tool import tool

from agent.realtime.tools.weather import get_weather_for_current_location
from agent.realtime.tools.web_search import run_web_search_agent
from audio.player.audio_manager import AudioManager

# provide more guidance in description when to call the tool and what the assistant should say when calling it


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
async def delegate_task_to_web_search_agent(
    query: Annotated[
        str, "The search query or task to delegate to the web search agent"
    ],
) -> str:
    """Perform a web search and return the aggregated results."""
    return await run_web_search_agent(query)


@tool(description="Play startup sound")
async def play_sound(audio_manager: AudioManager) -> None:
    audio_manager.get_strategy().play_startup_sound()


@tool(description="Adjust assistant volume level")
async def adjust_volume(
    level: Annotated[float, "Volume level from 0.0 (0%) to 1.0 (100%)"],
    audio_manager: AudioManager,
) -> None:
    audio_manager.get_strategy().set_volume_level(level)


# Tools for stopping agent run
# response speed verändern
# wake word verändern
