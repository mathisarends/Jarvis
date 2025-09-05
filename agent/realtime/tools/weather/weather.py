import asyncio
import aiohttp

from agent.realtime.tools.location.location import get_current_location
from agent.realtime.tools.weather.views import OpenMeteoApiResponse, WeatherData
from agent.realtime.tools.weather.formatting import (
    format_weather_report,
    transform_api_response,
)


async def fetch_weather_data(latitude: float, longitude: float) -> WeatherData:
    """Fetch weather data from Open-Meteo API and transform to domain models."""

    async with aiohttp.ClientSession() as session:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,pressure_msl",
            "hourly": "temperature_2m,weather_code,precipitation_probability,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum,wind_speed_10m_max",
            "timezone": "auto",
            "wind_speed_unit": "ms",
            "forecast_days": 3,
        }

        async with session.get(url, params=params) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Error while fetching weather data: {response.status}"
                )

            raw_data = await response.json()

            api_response = OpenMeteoApiResponse.model_validate(raw_data)
            weather_data = transform_api_response(api_response)

            return weather_data


async def get_weather_for_current_location() -> str:
    """Main function: Get location and fetch 3-day weather report."""

    try:
        location = await get_current_location()

        weather_data = await fetch_weather_data(location.latitude, location.longitude)

        return format_weather_report(weather_data, location)

    except Exception as e:
        return f"❌ Error: {e}"


async def test_weather():
    """Test the 3-day weather functionality."""

    print("🌤️ Loading current weather + 3-day forecast...")
    result = await get_weather_for_current_location()
    print(result)


if __name__ == "__main__":
    asyncio.run(test_weather())
