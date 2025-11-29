import asyncio

from agent import Agent
from agent.config import (
    ModelSettings,
    TranscriptionSettings,
    VoiceSettings,
    WakeWordSettings,
)
from agent.events.schemas import AssistantVoice, NoiseReductionType
from agent.wake_word import PorcupineWakeWord


async def main():
    model_settings = ModelSettings(
        instructions="Be concise and friendly. Answer in German. Always use tools if necessary.",
        temperature=0.8,
    )

    voice_settings = VoiceSettings(
        assistant_voice=AssistantVoice.MARIN,
        speech_speed=1.3,
    )

    transcription_settings = TranscriptionSettings(
        enabled=False,
        noise_reduction_mode=NoiseReductionType.NEAR_FIELD,
    )

    wake_word_settings = WakeWordSettings(
        enabled=True,
        keyword=PorcupineWakeWord.PICOVOICE,
        sensitivity=0.7,
    )

    try:
        agent = Agent(
            instructions="seit nett",
            model_settings=model_settings,
            voice_settings=voice_settings,
            transcription_settings=transcription_settings,
            wake_word_settings=wake_word_settings,
        )

        await agent.start()

    except KeyboardInterrupt:
        pass  # Graceful shutdown
    except Exception:
        print("Critical application error")
        raise


if __name__ == "__main__":
    asyncio.run(main())
