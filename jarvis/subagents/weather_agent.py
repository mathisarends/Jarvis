from datetime import datetime
from typing import Annotated

import httpx

from rtvoice import SubAgent, Tools
from llmify import ChatOpenAI


def _build_weather_tools() -> Tools:
    tools = Tools()

    @tools.action("Get the current local date and time.")
    def get_current_time() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @tools.action(
        "Detect the user's current location based on their IP address. "
        "Always call this first before fetching weather."
    )
    async def get_current_location() -> str:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://ipapi.co/json/")
            data = response.json()
            return f"{data['city']}, {data['region']}, {data['country_name']}"

    @tools.action(
        "Fetch current weather and hourly forecast for a given location. "
        "Returns current conditions plus an hourly breakdown for the next 48 hours "
        "so time-specific questions like 'this evening' or 'tomorrow afternoon' can be answered accurately."
    )
    async def get_weather(
        location: Annotated[str, "City name, e.g. 'Münster, Germany'"],
    ) -> str:
        async with httpx.AsyncClient() as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en"},
            )
            results = geo.json().get("results")
            if not results:
                return f"Location '{location}' not found."

            r = results[0]
            lat, lon = r["latitude"], r["longitude"]

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
                    f"  {hours[i]}: {hourly['temperature_2m'][i]}°C, "
                    f"rain chance {hourly['precipitation_probability'][i]}%"
                )

            return (
                f"Location: {r['name']}, {r['country_code']}\n\n"
                f"Current conditions:\n"
                f"  Temperature: {current['temperature_2m']}°C "
                f"(feels like {current['apparent_temperature']}°C)\n"
                f"  Precipitation: {current['precipitation']}mm\n\n"
                f"Hourly forecast (next 48h):\n"
                + "\n".join(hourly_lines)
            )

    return tools


_INSTRUCTIONS = (
    "You are a weather assistant.\n\n"
    "When asked about the weather, always follow these steps:\n"
    "1. Call get_current_location to determine the user's current location.\n"
    "2. Call get_weather with that location.\n"
    "3. Answer based on what the user actually asked:\n"
    "   - Asked in the morning (before 11:00) → give a brief day overview: "
    "how the temperature develops and whether it will rain and when.\n"
    "   - 'this evening' / 'tonight' → use the hourly forecast for 18:00–22:00\n"
    "   - 'this afternoon' → use 12:00–17:00\n"
    "   - 'tomorrow' → use tomorrow's hourly data\n"
    "   - general / current → current conditions only\n\n"
    "Never assume or hardcode a location – always fetch it fresh via get_current_location.\n\n"
    "Be extremely concise – 1 to 2 sentences max. "
    "Focus only on two things: temperature feel and rain. "
    "Say 'pretty warm', 'mild', 'cold' etc. "
    "For rain: mention if and roughly when it might rain. If no rain expected, say so briefly.\n\n"
    "When calling the done tool, always begin your answer by mentioning the city "
    "that was fetched dynamically via get_current_location."
)


def create_weather_agent() -> SubAgent:
    return SubAgent(
        name="Weather Agent",
        description=(
            "Provides current weather information and forecasts. "
            "Use this agent for questions about weather, temperature, rain, "
            "or conditions at specific times like 'this evening' or 'later today'."
        ),
        instructions=_INSTRUCTIONS,
        tools=_build_weather_tools(),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
        handoff_instructions="Use this agent for weather-related questions.",
    )