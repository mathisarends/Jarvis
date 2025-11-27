from datetime import datetime
from typing import Annotated

from agent.config.models import VoiceSettings
from agent.realtime.event_bus import EventBus
from agent.realtime.events.client.session_update import MCPTool
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.assistant import run_volume_adjustment_agent
from agent.state.base import VoiceAssistantEvent
from agent.sound import AudioPlayer
from shared.logging_mixin import LoggingMixin


class Tools(LoggingMixin):
    def __init__(
        self,
        mcp_tools: list[MCPTool] | None = None,
    ):
        self._mcp_tools = mcp_tools
        self.registry = ToolRegistry(mcp_tools=mcp_tools)
        self._register_default_tools()

    @property
    def mcp_tools(self) -> list[MCPTool] | None:
        return self._mcp_tools

    @mcp_tools.setter
    def mcp_tools(self, mcp_tools: list[MCPTool] | None) -> None:
        self._mcp_tools = mcp_tools

        # Sync with registry
        self.registry.mcp_tools = mcp_tools or []
        self.logger.info(
            "MCP tools updated: %d tools", len(mcp_tools) if mcp_tools else 0
        )

    def action(self, description: str, **kwargs):
        return self.registry.action(description, **kwargs)

    def _register_default_tools(self) -> None:
        @self.registry.action("Get the current local time")
        def get_current_time() -> str:
            return datetime.now().strftime("%H:%M:%S")

        @self.registry.action("Adjust volume level.")
        def adjust_volume(
            level: Annotated[float, "Volume level from 0.0 (0%) to 1.0 (100%)"],
            audio_player: AudioPlayer,
        ) -> None:
            audio_player.set_volumne_level(level)

        @self.registry.action(
            "Change the assistant's talking speed by a relative amount. Acknowledge the change before calling the tool. The tool internally retrieves the current speed and adjusts it relative to the current rate.",
            execution_message="Adjusting response speed...",
            response_instruction="State that the response speed has been adjusted and name the new speed in percent (e.g. 1.5 = 150%)",
        )
        async def change_assistant_response_speed(
            instructions: Annotated[
                str, "Natural language command: 'faster' or 'slower'"
            ],
            voice_settings: VoiceSettings,
            event_bus: EventBus,
        ) -> str:
            current_response_speed = voice_settings.speech_speed
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
