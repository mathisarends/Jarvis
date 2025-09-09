from datetime import datetime
from typing import Annotated

from agent.config.views import AgentConfig
from agent.realtime.event_bus import EventBus
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.weather import get_weather_for_current_location
from agent.realtime.tools.web_search import run_web_search_agent
from agent.realtime.tools.assistant import run_volume_adjustment_agent
from agent.state.base import VoiceAssistantEvent
from audio.player.audio_manager import AudioManager
from shared.logging_mixin import LoggingMixin


class Tools(LoggingMixin):
    def __init__(
        self,
    ):
        self.registry = ToolRegistry()
        self._register_default_tools()

    def action(self, description: str, **kwargs):
        return self.registry.action(description, **kwargs)

    def _register_default_tools(self) -> None:
        """Register all default tools using the action decorator."""

        @self.registry.action("Get the current local time")
        def get_current_time() -> str:
            return datetime.now().strftime("%H:%M:%S")

        @self.registry.action(
            "Get weather forecast for your current location. Automatically detects your location via IP and fetches detailed weather data including current conditions and forecasts.",
            response_instruction="State the number of forecast days: 1 for today only, 3 for 3-day forecast (default: 1).",
        )
        async def get_weather(
            forecast_days: Annotated[
                int,
                "Number of forecast days: 1 for today only, 3 for 3-day forecast (default: 1)",
            ] = 1,
        ) -> str:
            """Get weather report for current location with configurable forecast days."""
            return await get_weather_for_current_location(forecast_days)

        @self.registry.action(
            "Delegates a task to a specialized web search agent that automatically optimizes the query with contextual information and returns aggregated search results from the web. Use this whenever the user asks for information that requires up-to-date knowledge or specific details from the web."
        )
        async def delegate_task_to_web_search_agent(
            query: Annotated[
                str, "The search query or task to delegate to the web search agent"
            ],
        ) -> str:
            """Perform a web search and return the aggregated results."""
            return await run_web_search_agent(query)

        @self.registry.action("Adjust volume level.")
        def adjust_volume(
            level: Annotated[float, "Volume level from 0.0 (0%) to 1.0 (100%)"],
            audio_manager: AudioManager,
        ) -> None:
            audio_manager.strategy.set_volume_level(level)

        @self.registry.action(
            "Change the assistant's talking speed by a relative amount. Acknowledge the change before calling the tool. The tool internally retrieves the current speed and adjusts it relative to the current rate.",
            execution_message="Adjusting response speed...",
            response_instruction="State that the response speed has been adjusted and name the new speed in percent (e.g. 1.5 = 150%)",
        )
        async def change_assistant_response_speed(
            instructions: Annotated[
                str, "Natural language command: 'faster' or 'slower'"
            ],
            agent_config: AgentConfig,
            event_bus: EventBus,
        ) -> str:
            current_response_speed = agent_config.speed
            response_speed_adjustment_result = await run_volume_adjustment_agent(
                instructions, current_response_speed
            )
            new_response_speed = response_speed_adjustment_result.new_response_speed

            await event_bus.publish_async(
                VoiceAssistantEvent.ASSISTANT_CONFIG_UPDATE_REQUEST, new_response_speed
            )

            return f"Volume adjusted to {new_response_speed*100:.0f}%"

        @self.registry.action(
            "Stop the assistant run. Call this when the user says 'stop', 'cancel', or 'abort'."
        )
        async def stop_assistant_run(event_bus: EventBus) -> None:
            await event_bus.publish_async(VoiceAssistantEvent.IDLE_TRANSITION)
