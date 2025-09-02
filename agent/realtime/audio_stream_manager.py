import asyncio
from typing import Any

from agent.realtime.websocket_manager import WebSocketManager
from audio.capture import AudioCapture
from audio.sound_player import SoundPlayer
from shared.logging_mixin import LoggingMixin


class AudioStreamManager(LoggingMixin):
    """
    Manages audio streaming for real-time communication with OpenAI API.
    Handles both sending microphone audio and receiving/playing assistant responses.
    """

    def __init__(self, ws_manager: WebSocketManager, sound_player: SoundPlayer, audio_capture: AudioCapture):
        self.ws_manager = ws_manager
        self.sound_player = sound_player
        self.audio_capture = audio_capture
        self.logger.info("AudioStreamManager initialized")

    async def send_audio_stream(self) -> None:
        """
        Sends audio data from the microphone to the OpenAI API.
        """
        if not self.ws_manager.is_connected():
            self.logger.error("No connection available for audio transmission")
            return

        try:
            self.logger.info("Starting audio transmission...")
            audio_chunks_sent = 0

            while self.audio_capture.is_active and self.ws_manager.is_connected():
                data = self.audio_capture.read_chunk()
                if not data:
                    await asyncio.sleep(0.01)
                    continue

                success = await self.ws_manager.send_binary(data)
                if success:
                    audio_chunks_sent += 1
                    if audio_chunks_sent % 100 == 0:
                        self.logger.debug("Audio chunks sent: %d", audio_chunks_sent)
                else:
                    self.logger.warning("Failed to send audio chunk")

                await asyncio.sleep(0.01)

        except asyncio.TimeoutError as e:
            self.logger.error("Timeout while sending audio: %s", e)
        except Exception as e:
            self.logger.error("Error while sending audio: %s", e)

    def enqueue_audio_chunk(self, response: dict[str, Any]) -> None:
        """
        Processes audio responses from the OpenAI API.
        """
        base64_audio = response.get("delta", "")
        if not base64_audio or not isinstance(base64_audio, str):
            return

        self.sound_player.add_audio_chunk(base64_audio)

    def stop_playback(self) -> None:
        """
        Stops audio playback.
        """
        self.sound_player.clear_queue_and_stop_chunks()
