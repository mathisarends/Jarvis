import logging

from hueify import Hueify
from rtvoice import SubAgent
from rtvoice.mcp import MCPServerStdio

logger = logging.getLogger(__name__)


def _build_instructions(hueify: Hueify) -> str:
    lights = hueify.lights.names
    rooms = hueify.rooms.names
    zones = hueify.zones.names

    sections = [
        "You are a smart home lighting assistant with access to Philips Hue lights via MCP tools.",
        "When the user asks you to control lights, use the available tools to do so.",
        "Confirm what you did in one short sentence. No follow-up offers or questions.",
        "",
        "## Available Resources",
        "",
        f"**Individual lights:** {', '.join(lights) if lights else 'none'}",
        f"**Rooms:** {', '.join(rooms) if rooms else 'none'}",
        f"**Zones:** {', '.join(zones) if zones else 'none'}",
        "",
        "Prefer controlling rooms or zones over individual lights unless the user specifically asks for a single light.",
    ]
    return "\n".join(sections)


async def create_light_agent(llm) -> SubAgent:
    async with Hueify() as hueify:
        instructions = _build_instructions(hueify)

    logger.debug(
        "Hueify context closed – MCP server will establish its own connection at runtime"
    )

    return SubAgent(
        name="LightAgent",
        description=(
            "Controls Philips Hue lights, rooms, and zones. "
            "Use for any lighting request: on/off, brightness, color temperature, scenes."
        ),
        instructions=instructions,
        mcp_servers=[
            MCPServerStdio(
                command="uv",
                args=["run", "-m", "hueify.mcp.server"],
                allowed_tools=[
                # Lights
                "turn_on_light",
                "turn_off_light",
                "set_light_brightness",
                "increase_light_brightness",
                "decrease_light_brightness",
                "get_light_brightness",
                "list_lights",
                # Rooms
                "turn_on_room",
                "turn_off_room",
                "set_room_brightness",
                "increase_room_brightness",
                "decrease_room_brightness",
                "get_room_brightness",
                "activate_scene_in_room",
                "get_active_scene_in_room",
                "list_scenes_in_room",
                "list_rooms",
                ],
            ),
        ],
        llm=llm,
        handoff_instructions=(
            "Use this agent when the user wants to control lights, "
            "change brightness or color temperature, activate scenes, "
            "or turn lights/rooms/zones on or off."
        ),
        result_instructions=(
            "Return a short, natural confirmation of what was changed. "
            "No follow-up offers, no 'Sag Bescheid', no questions."
        ),
    )