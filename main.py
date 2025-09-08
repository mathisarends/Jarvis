"""
Voice Assistant Application Entry Point.
"""

import asyncio
import os
from typing import Optional

from agent.agent import RealtimeAgent
from agent.realtime.events.client.session_update import NoiseReductionType
from agent.realtime.views import AssistantVoice
from audio.wake_word_listener import PorcupineBuiltinKeyword


class VoiceAssistantApp:
    """Application entry point with proper lifecycle management."""

    def __init__(self):
        self._agent: Optional[RealtimeAgent] = None

    async def run(self) -> None:
        """Run the voice assistant application."""
        try:
            # Create agent with configuration from YAML
            self._agent = RealtimeAgent(
                instructions="Be concise and friendly. Answer in German.",
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
            
            # Run the agent
            await self._agent.run()

        except KeyboardInterrupt:
            pass  # Graceful shutdown
        except Exception:
            print("Critical application error")
            raise

    async def _shutdown(self) -> None:
        """Graceful application shutdown."""
        # Agent handles its own shutdown
        pass


async def main():
    """Application entry point."""
    app = VoiceAssistantApp()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
