"""
Voice Assistant Application Entry Point.
"""

import asyncio

from agent.agent import RealtimeAgent
from agent.realtime.events.client.session_update import NoiseReductionType
from agent.realtime.views import AssistantVoice
from audio.wake_word_listener import PorcupineBuiltinKeyword


async def main():
    """Run the voice assistant application."""
    try:
        agent = RealtimeAgent(
            instructions="Be concise and friendly. Answer in German. Always use tools if necessary.",
            response_temperature=0.8,
            assistant_voice=AssistantVoice.MARIN,
            speech_speed=1.3,
            enable_transcription=True,
            transcription_language="de",
            noise_reduction_mode=NoiseReductionType.NEAR_FIELD,
            enable_wake_word=True,
            wakeword=PorcupineBuiltinKeyword.PICOVOICE,
            wake_word_sensitivity=0.7,
        )

        await agent.run()

    except KeyboardInterrupt:
        pass  # Graceful shutdown
    except Exception:
        print("Critical application error")
        raise


if __name__ == "__main__":
    asyncio.run(main())
