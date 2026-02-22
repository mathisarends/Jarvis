import asyncio
import logging

from jarvis import Jarvis
from jarvis.subagents import WeatherAgent

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    tools = None
    weather_agent = WeatherAgent()

    jarvis = Jarvis(
        subagents=[weather_agent],
    )
    await jarvis.run()

if __name__ == "__main__":
    asyncio.run(main())