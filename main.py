import asyncio

from rtvoice import AssistantVoice
from rtvoice.views import NoiseReduction
from jarvis import Jarvis, WakeWord, configure_logging
from jarvis.subagents import WeatherAgent, LightAgent

configure_logging()

async def main() -> None:
    weather_agent = WeatherAgent()
    light_agent = LightAgent()

    instructions = (
        "Du bist Jarvis, ein persönlicher Sprachassistent. "
        "Antworte immer auf Deutsch, kurz und direkt – maximal 1–2 Sätze. "
        "Keine Rückfragen, keine Höflichkeitsfloskeln wie 'Kann ich sonst noch helfen?' oder 'Gerne unterstütze ich Sie'. "
    )

    jarvis = Jarvis(
        voice=AssistantVoice.ECHO,
        wake_word=WakeWord.PICOVOICE,
        subagents=[weather_agent, light_agent],
        instructions=instructions,
        noise_reduction=NoiseReduction.FAR_FIELD
    )
    await jarvis.run()


if __name__ == "__main__":
    asyncio.run(main())