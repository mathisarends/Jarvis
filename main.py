import asyncio
import sys

from rtvoice import AssistantVoice
from rtvoice.views import NoiseReduction
from rtvoice.mcp import MCPServerStdio
from jarvis import Jarvis, WakeWord, configure_logging
from jarvis.subagents import WeatherAgent
""" from jarvis.sonos import SonosAudioOutputDevice """

configure_logging()


async def main() -> None:
    weather_agent = WeatherAgent()

    hueify_mcp = MCPServerStdio(
        command=sys.executable,
        args=["-m", "hueify.mcp.app"],
    )

    instructions = (
        "Du bist Jarvis, ein persönlicher Sprachassistent. "
        "Antworte immer auf Deutsch, kurz und direkt – maximal 1–2 Sätze. "
        "Keine Rückfragen, keine Höflichkeitsfloskeln wie 'Kann ich sonst noch helfen?' oder 'Gerne unterstütze ich Sie'. "
        "Erfinde niemals Kontext über den Nutzer, seine Umgebung oder seinen Zustand – "
        "mache keine Aussagen über Gegenstände, Lichter, Kleidung oder andere Details, "
        "die du nicht explizit durch ein Tool-Ergebnis oder den Nutzer selbst erfahren hast. "
        "Halte dich strikt an das, was dir bekannt ist."
    )

    jarvis = Jarvis(
        voice=AssistantVoice.MARIN,
        wake_word=WakeWord.PICOVOICE,
        subagents=[weather_agent],
        mcp_servers=[hueify_mcp],
        instructions=instructions,
        noise_reduction=NoiseReduction.NEAR_FIELD,
    )
    await jarvis.run()


if __name__ == "__main__":
    asyncio.run(main())