import asyncio
from datetime import datetime
from typing import AsyncGenerator
from agent.realtime.tools.tool import tool

from agent.realtime.tools.weather import get_weather_for_current_location


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
    description="Simuliert eine Browser-Automation und streamt Statusupdates einer Recherche.",
    result_context="Stellt die einzelnen Schritte einer Online-Recherche dar.",
)
async def stream_browser_search(topic: str) -> AsyncGenerator[str, None]:
    """Führt eine simulierte Browser-Recherche zu einem Thema durch und streamt den Fortschritt."""

    # Natürlicher, menschlicher formuliert
    yield "Okay, ich öffne jetzt den Browser"
    await asyncio.sleep(8)

    yield f"Ich suche gerade nach aktuellen Informationen zu {topic}"
    await asyncio.sleep(8)

    yield f"Super, ich finde schon erste interessante Artikel zu {topic}"
    await asyncio.sleep(8)

    yield f"Ich schaue mir gerade die neuesten Nachrichten zu {topic} an"
    await asyncio.sleep(8)

    yield "Ich erweitere die Suche um verwandte Themen"
    await asyncio.sleep(8)

    yield "Perfekt, die Recherche ist abgeschlossen!"
