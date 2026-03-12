from datetime import datetime
from typing import Annotated

import httpx
from hueify import Hueify
from rtvoice.mcp import MCPServerStdio
from rtvoice.supervisor import SupervisorAgent
from rtvoice.tools import SupervisorTools


async def _get_user_location() -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get("https://ipapi.co/json/")
        data = response.json()
        return f"{data['city']}, {data['region']}, {data['country_name']}"


def _build_tools() -> SupervisorTools:
    tools = SupervisorTools()

    @tools.action("Get the current local date and time.")
    def get_current_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @tools.action(
        "Detect the user's current location based on their IP address. "
        "Always call this first before fetching weather."
    )
    async def get_current_location() -> str:
        return await _get_user_location()

    @tools.action(
        "Fetch current weather and hourly forecast for a given location. "
        "Returns current conditions plus an hourly breakdown for the next 48 hours "
        "so time-specific questions like 'this evening' or 'tomorrow afternoon' can be answered accurately."
    )
    async def get_weather(
        location: Annotated[str, "City name, e.g. 'Muenster, Germany'"],
    ) -> str:
        async with httpx.AsyncClient() as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en"},
            )
            results = geo.json().get("results")
            if not results:
                return f"Location '{location}' not found."

            result = results[0]
            lat, lon = result["latitude"], result["longitude"]

            weather = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,apparent_temperature,precipitation,weathercode",
                    "hourly": "temperature_2m,apparent_temperature,precipitation_probability,precipitation,weathercode",
                    "forecast_days": 2,
                    "timezone": "Europe/Berlin",
                },
            )
            data = weather.json()
            current = data["current"]
            hourly = data["hourly"]

            hours = hourly["time"]
            hourly_lines = []
            for i in range(0, len(hours), 3):
                hourly_lines.append(
                    f"  {hours[i]}: {hourly['temperature_2m'][i]} degC, "
                    f"rain chance {hourly['precipitation_probability'][i]}%"
                )

            return (
                f"Location: {result['name']}, {result['country_code']}\\n\\n"
                f"Current conditions:\\n"
                f"  Temperature: {current['temperature_2m']} degC "
                f"(feels like {current['apparent_temperature']} degC)\\n"
                f"  Precipitation: {current['precipitation']}mm\\n\\n"
                f"Hourly forecast (next 48h):\\n"
                + "\\n".join(hourly_lines)
            )

    return tools


def _build_instructions(location: str, lights: list[str], rooms: list[str], zones: list[str]) -> str:
    return (
        "You are Jarvis's supervisor agent. You can handle weather and Philips Hue lighting tasks.\\n\\n"
        f"The user's current location is: {location}\\n\\n"
        "Weather behavior:\\n"
        "- Use this location by default unless the user explicitly asks for another city.\\n"
        "- Answer based on requested time window (current, afternoon, evening, tomorrow).\\n"
        "- Keep weather results concise and focused on temperature feel and rain.\\n\\n"
        "Lighting behavior:\\n"
        "- Use MCP Hue tools to control lights.\\n"
        "- Prefer rooms/zones over individual lights unless the user asks for a specific light.\\n"
        "- Confirm the performed action in one short sentence.\\n\\n"
        "Available Hue resources:\\n"
        f"- Lights: {', '.join(lights) if lights else 'none'}\\n"
        f"- Rooms: {', '.join(rooms) if rooms else 'none'}\\n"
        f"- Zones: {', '.join(zones) if zones else 'none'}"
    )


async def create_supervisor_agent(llm) -> SupervisorAgent:
    location = await _get_user_location()

    async with Hueify() as hueify:
        instructions = _build_instructions(
            location=location,
            lights=hueify.lights.names,
            rooms=hueify.rooms.names,
            zones=hueify.zones.names,
        )

    return SupervisorAgent(
        name="Jarvis Supervisor",
        description=(
            "Handles weather requests and Philips Hue lighting control. "
            "Use this agent for forecasts, temperature/rain questions, and light operations "
            "like on/off, brightness, scenes, and room controls."
        ),
        instructions=instructions,
        tools=_build_tools(),
        llm=llm,
        mcp_servers=[
            MCPServerStdio(
                command="uv",
                args=["run", "-m", "hueify.mcp.server"],
                allowed_tools=[
                    "turn_on_light",
                    "turn_off_light",
                    "set_light_brightness",
                    "increase_light_brightness",
                    "decrease_light_brightness",
                    "get_light_brightness",
                    "list_lights",
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
        handoff_instructions=(
            "Use this agent when the user asks about weather or controlling Hue lights."
        ),
        result_instructions=(
            "Return only the concrete result in 1-2 short sentences without follow-up questions."
        ),
    )