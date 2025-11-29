import asyncio

from rtvoice import Agent
from rtvoice.events.schemas import AssistantVoice, NoiseReductionType
from rtvoice.wake_word import PorcupineWakeWord


async def main():
    try:
        agent = Agent(
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
