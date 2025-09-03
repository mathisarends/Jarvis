from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent.realtime.event_types import RealtimeClientEvent
from agent.realtime.transcription.service import TranscriptionService
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.realtime.views import (
    SessionUpdateEvent,
    AudioConfig,
    AudioOutputConfig,
    AudioFormatConfig,
    AudioFormat,
    SessionConfig,
    RealtimeModel,
)
from audio.capture import AudioCapture
from shared.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from agent.realtime.views import VoiceAssistantConfig


class OpenAIRealtimeAPI(LoggingMixin):

    def __init__(
        self,
        realtime_config: VoiceAssistantConfig,
        ws_manager: WebSocketManager,
        audio_capture: AudioCapture,
        transcription_service: TranscriptionService,
    ):
        """
        Initializes the OpenAI Realtime API client.
        All configuration is loaded from configuration files.
        """
        self.ws_manager = ws_manager
        self.audio_capture = audio_capture
        self.transcription_service = transcription_service

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
            # Only send audio stream - WebSocketManager handles all responses automatically
            await self._send_audio_stream()
            return True
        except asyncio.CancelledError:  # NOSONAR
            self.logger.info("Audio streaming was cancelled")
            return True
        finally:
            await self.ws_manager.close()

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
        Uses fully typed Pydantic models based on the official API documentation.
        """
        # Create audio configuration with nested format objects
        audio_config = AudioConfig(
            output=AudioOutputConfig(
                format=AudioFormatConfig(type=AudioFormat.PCM16),
                voice=self.voice or "marin",
            )
        )

        # Create session configuration using typed models
        session_config = SessionUpdateEvent(
            type=RealtimeClientEvent.SESSION_UPDATE,
            session=SessionConfig(
                type="realtime",
                model=RealtimeModel.GPT_REALTIME,
                instructions=self.system_message,
                audio=audio_config,
                output_modalities=["audio"],
                max_output_tokens=1024,
            ),
        )

        # Return the validated configuration as dict
        return session_config.model_dump(exclude_unset=True)

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

                success = await self.ws_manager.send_audio_stream(data)
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
