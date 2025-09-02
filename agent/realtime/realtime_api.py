import asyncio
from dataclasses import dataclass
from agent.realtime.audio_stream_manager import AudioStreamManager
from agent.realtime.websocket_manager import WebSocketManager
from agent.realtime.message_manager import RealtimeSessionMessageManager
from audio.capture import AudioCapture
from audio.sound_player import SoundPlayer
from shared.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class OpenAIRealtimeAPIConfig:
    system_message: str
    voice: str
    temperature: float


class OpenAIRealtimeAPI(LoggingMixin):

    def __init__(
        self,
        realtime_config: OpenAIRealtimeAPIConfig,
        ws_manager: WebSocketManager,
        sound_player: SoundPlayer,
        audio_capture: AudioCapture,
    ):
        """
        Initializes the OpenAI Realtime API client.
        All configuration is loaded from configuration files.
        """
        self.ws_manager = ws_manager
        self.sound_player = sound_player
        self.audio_capture = audio_capture

        self.audio_handler = AudioStreamManager(ws_manager=self.ws_manager, sound_player=self.sound_player, audio_capture=self.audio_capture)

        self.session_manager = RealtimeSessionMessageManager(
            ws_manager=self.ws_manager, config=realtime_config
        )

    async def setup_and_run(self) -> bool:
        """
        Sets up the connection and runs the main loop.
        """
        if not await self.ws_manager.create_connection():
            return False

        if not await self.session_manager.initialize_session():
            await self.ws_manager.close()
            return False

        try:
            await asyncio.gather(
                self.audio_handler.send_audio_stream(),
                self._process_responses(),
            )

            return True
        except asyncio.CancelledError: # NOSONAR
            self.logger.info("Tasks were cancelled")
            return True
        finally:
            await self.ws_manager.close()
            
    async def _process_responses(self) -> None:
        """
        Processes responses from the OpenAI API and publishes events to the EventBus.
        """
        if not self.ws_manager.is_connected():
            self.logger.error("No connection available for processing responses")
            return

        await self.ws_manager.receive_messages(
            should_continue=self.ws_manager.is_connected,
        )
