"""
Voice Assistant Application Entry Point.
"""

import asyncio
from typing import Optional

from agent.config.env import validate_environment_variables
from agent.config.loader import load_config_from_yaml
from agent.controller.voice_assistant_controller import VoiceAssistantController


class VoiceAssistantApp:
    """Application entry point with proper lifecycle management."""

    def __init__(self):
        self._controller: Optional[VoiceAssistantController] = None

    async def run(self) -> None:
        """Run the voice assistant application."""
        try:
            # Setup
            validate_environment_variables()
            config = load_config_from_yaml()

            # Run
            self._controller = VoiceAssistantController(config)
            await self._controller.start()

        except KeyboardInterrupt:
            pass  # Graceful shutdown
        except Exception:
            print("Critical application error")
            raise
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """Graceful application shutdown."""
        if self._controller:
            await self._controller.stop()


async def main():
    """Application entry point."""
    app = VoiceAssistantApp()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
