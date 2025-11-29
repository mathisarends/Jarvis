import asyncio

from rtvoice import Agent, Tools
from rtvoice.events.schemas import AssistantVoice, NoiseReductionType
from rtvoice.wake_word import PorcupineWakeWord

tools = Tools()


@tools.action(
    name="get_current_weather",
    description="Get the current weather for a given location.",
    response_instruction="Provide a brief summary of the current weather.",
)
def get_current_weather(location: str) -> str:
    return f"The current weather in {location} is sunny with a temperature of 25°C."


async def main():
    try:
        agent = Agent(
            tools=tools,
            instructions="Be concise and friendly. Answer in German. Always use tools if necessary.",
            voice=AssistantVoice.MARIN,
            speech_speed=1.3,
            temperature=0.8,
            enable_transcription=False,
            noise_reduction=NoiseReductionType.NEAR_FIELD,
            wake_word=PorcupineWakeWord.PICOVOICE,
            wake_word_sensitivity=0.7,
        )

        await agent.start()

    except KeyboardInterrupt:
        pass
    except Exception:
        print("Critical application error")
        raise


if __name__ == "__main__":
    asyncio.run(main())
