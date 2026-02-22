import asyncio

from rtvoice import AssistantVoice
from rtvoice.views import NoiseReduction
from jarvis import Jarvis, WakeWord, configure_logging
from jarvis.subagents import WeatherAgent, LightAgent
from jarvis.sonos import SonosAudioOutputDevice

configure_logging()


async def main() -> None:
    weather_agent = WeatherAgent()
    light_agent = LightAgent()

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
        subagents=[weather_agent, light_agent],
        instructions=instructions,
        noise_reduction=NoiseReduction.NEAR_FIELD,
        audio_output_device=SonosAudioOutputDevice(),
    )
    await jarvis.run()


if __name__ == "__main__":
    asyncio.run(main())