from typing import Annotated

import httpx

from rtvoice import SubAgent, Tools
from llmify import ChatOpenAI


class WeatherAgent(SubAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Weather Agent",
            description=(
                "Provides current weather information and forecasts. "
                "Use this agent for questions about weather, temperature, rain, wind, "
                "or conditions at specific times like 'this evening' or 'later today'."
            ),
            instructions=(
                "You are a weather assistant. When asked about the weather:\n"
                "1. Call get_location to determine where the user is.\n"
                "2. Call get_weather with that location.\n"
                "3. Answer based on what the user actually asked:\n"
                "   - 'right now' / 'today' → use current conditions\n"
                "   - 'this evening' / 'tonight' → use the hourly forecast for 18:00–22:00\n"
                "   - 'this afternoon' → use 12:00–17:00\n"
                "   - 'tomorrow' → use tomorrow's hourly data\n"
                "   - general overview → summarize the day's trend\n\n"
                "Be concise and conversational. Don't read raw numbers – "
                "say 'pretty warm' or 'quite windy' where appropriate. "
                "Mention rain chances if relevant."
            ),
            tools=self._build_tools(),
            llm=ChatOpenAI(model="gpt-4o", temperature=0.2),
            handoff_instructions="Use this agent for weather-related questions.",
            pending_message="Let me check the weather for you...",
        )

    def _build_tools(self) -> Tools:
        tools = Tools()

        @tools.action("Get the user's current location based on IP geolocation.")
        async def get_location() -> str:
            async with httpx.AsyncClient() as client:
                response = await client.get("https://ipapi.co/json/")
                data = response.json()
                return f"{data['city']}, {data['region']}, {data['country_name']}"

        @tools.action(
            "Fetch current weather and hourly forecast for a given location. "
            "Always call get_location first. Returns current conditions plus "
            "an hourly breakdown for the next 48 hours so time-specific questions "
            "like 'this evening' or 'tomorrow afternoon' can be answered accurately."
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
                        "current": "temperature_2m,apparent_temperature,precipitation,windspeed_10m,weathercode",
                        "hourly": "temperature_2m,apparent_temperature,precipitation_probability,precipitation,windspeed_10m,weathercode",
                        "forecast_days": 2,
                        "wind_speed_unit": "kmh",
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
                        f"rain chance {hourly['precipitation_probability'][i]}%, "
                        f"wind {hourly['windspeed_10m'][i]} km/h"
                    )

                return (
                    f"Location: {r['name']}, {r['country_code']}\n\n"
                    f"Current conditions:\n"
                    f"  Temperature: {current['temperature_2m']}°C "
                    f"(feels like {current['apparent_temperature']}°C)\n"
                    f"  Precipitation: {current['precipitation']}mm\n"
                    f"  Wind: {current['windspeed_10m']} km/h\n\n"
                    f"Hourly forecast (next 48h):\n"
                    + "\n".join(hourly_lines)
                )

        return tools