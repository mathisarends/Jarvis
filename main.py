import asyncio
import logging

from llmify import ChatOpenAI
from rtvoice import AssistantVoice, Tools
from rtvoice.views import NoiseReduction
from jarvis import Jarvis, WakeWord, configure_logging
from jarvis.subagents import create_light_agent, create_weather_agent

configure_logging()

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("hueify").setLevel(logging.WARNING)

async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    weather_agent = await create_weather_agent()
    light_agent = await create_light_agent(llm=llm)

    tools = Tools()

    instructions = (
        "Du bist Jarvis. Antworte auf Deutsch, maximal 1–2 Sätze, keine Floskeln, keine Rückfragen. "
        "Sag nur, was du durch ein Tool-Ergebnis oder den Nutzer selbst weißt – nichts erfinden."
    )
    
    jarvis = Jarvis(
        voice=AssistantVoice.MARIN,
        wake_word=WakeWord.HEY_JARVIS,
        subagents=[weather_agent, light_agent],
        tools=tools,
        instructions=instructions,
        noise_reduction=NoiseReduction.FAR_FIELD,
    )
    await jarvis.prepare()
    await jarvis.run()


if __name__ == "__main__":
    asyncio.run(main())
