"""
Voice Assistant Application Entry Point.
"""

import asyncio

from agent.agent import RealtimeAgent
from agent.realtime.events.client.session_update import NoiseReductionType
from agent.realtime.views import AssistantVoice
from audio.wake_word_listener import PorcupineBuiltinKeyword


# Beispiel für einen einfachen Context-Typ (hier ein Dict, aber es könnte jede Klasse sein)
class MyCustomContext:
    def __init__(self, user_name: str, session_id: str) -> None:
        self.user_name = user_name
        self.session_id = session_id

    def __str__(self):
        return f"Context(user={self.user_name}, session={self.session_id})"


async def main():
    """Run the voice assistant application."""

    custom_context = MyCustomContext(user_name="Alice", session_id="session_123")

    try:
        agent = RealtimeAgent[custom_context](
            instructions="Be concise and friendly. Answer in German. Always use tools if necessary.",
            response_temperature=0.8,
            assistant_voice=AssistantVoice.MARIN,
            speech_speed=1.3,
            enable_transcription=False,
            noise_reduction_mode=NoiseReductionType.NEAR_FIELD,
            enable_wake_word=True,
            wakeword=PorcupineBuiltinKeyword.PICOVOICE,
            wake_word_sensitivity=0.7,
        )

        # Der Agent kann den Context nun verwenden, z.B. in self.context.user_name zugreifen
        print(f"Agent started with context: {agent.context}")

        await agent.run()

    except KeyboardInterrupt:
        pass  # Graceful shutdown
    except Exception:
        print("Critical application error")
        raise


if __name__ == "__main__":
    asyncio.run(main())
