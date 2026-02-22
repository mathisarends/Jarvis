import asyncio
import logging

from jarvis import Jarvis

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    jarvis = Jarvis()
    await jarvis.run()


async def hueify() -> None:
    from hueify import Light

    light = await Light.from_name("Hue lightstrip plus 1")
    await light.turn_on()


if __name__ == "__main__":
    asyncio.run(main())