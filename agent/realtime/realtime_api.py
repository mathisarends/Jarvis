from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent.realtime.websocket_manager import WebSocketManager
from audio.capture import AudioCapture
from audio.sound_player import SoundPlayer
from shared.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from agent.realtime.views import VoiceAssistantConfig


class OpenAIRealtimeAPI(LoggingMixin):

    def __init__(
        self,
        realtime_config: VoiceAssistantConfig,
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

        # Session configuration from realtime_config
        self.system_message = realtime_config.system_message
        self.voice = realtime_config.voice
        self.temperature = realtime_config.temperature

    async def setup_and_run(self) -> bool:
        """
        Sets up the connection and runs the main loop.
        """
        if not await self.ws_manager.create_connection():
            return False

        if not await self._initialize_session():
            await self.ws_manager.close()
            return False

        try:
            await asyncio.gather(
                self._send_audio_stream(),
                self._process_responses(),
            )

            return True
        except asyncio.CancelledError:  # NOSONAR
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

    async def _initialize_session(self) -> bool:
        """
        Initializes a session with the OpenAI API.
        """
        if not self.ws_manager.is_connected():
            self.logger.error("No connection available for session initialization")
            return False

        session_update = self._build_session_config()

        try:
            self.logger.info("Sending session update...")
            success = await self.ws_manager.send_message(session_update)

            if success:
                self.logger.info("Session update sent successfully")
                return True

            self.logger.error("Failed to send session update")
            return False

        except Exception as e:
            self.logger.error("Error initializing session: %s", e)
            return False

    def _build_session_config(self) -> dict[str, Any]:
        """
        Creates the session configuration for the OpenAI API.
        """
        return {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": self.voice,
                "instructions": self.system_message,
                "modalities": ["audio"],
                "temperature": self.temperature,
                "tools": [],
            },
        }

    async def _send_audio_stream(self) -> None:
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
