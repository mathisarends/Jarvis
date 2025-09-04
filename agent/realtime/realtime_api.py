from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent.realtime.message_manager import RealtimeMessageManager
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.tool_executor import ToolExecutor
from agent.realtime.tools.tools import get_current_time
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

        # instantiate message manager (handles events and websocket messages)
        self.message_manager = RealtimeMessageManager(self.ws_manager)

        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)

        # Audio streaming control
        self._audio_streaming_paused = False
        self._audio_streaming_event = asyncio.Event()
        self._audio_streaming_event.set()  # Initially not paused

        self._register_tools()

    # this should not be possible in lifecycle (start -> setup_and_run -> close) | maybe refactor this
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

    async def close_connection(self) -> None:
        """Closes WebSocket and unblocks any waiting audio streams"""
        try:
            self.logger.info("Closing WebSocket connection programmatically...")

            # 1. WebSocket schließen
            await self.ws_manager.close()

            # 2. Audio-Event setzen um blockierten Task zu befreien
            self._audio_streaming_event.set()

            self.logger.info("WebSocket connection closed successfully")
        except Exception as e:
            self.logger.error("Error closing WebSocket connection: %s", e)

    def pause_audio_streaming(self) -> None:
        """
        Pauses the audio streaming. The WebSocket connection remains intact.
        """
        self.logger.info("Pausing audio streaming...")
        self._audio_streaming_paused = True
        self._audio_streaming_event.clear()

    def resume_audio_streaming(self) -> None:
        """
        Resumes the audio streaming.
        """
        self.logger.info("Resuming audio streaming...")
        self._audio_streaming_paused = False
        self._audio_streaming_event.set()

    def is_audio_streaming_paused(self) -> bool:
        """
        Returns whether the audio streaming is currently paused.
        """
        return self._audio_streaming_paused

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
                voice="marin",
            )
        )

        # Create session configuration using typed models
        session_config = SessionUpdateEvent(
            type="session.update",  # RealtimeClientEvent.SESSION_UPDATE
            session=SessionConfig(
                type="realtime",
                model=RealtimeModel.GPT_REALTIME,
                instructions=self.system_message,
                audio=audio_config,
                output_modalities=["audio"],
                max_output_tokens=1024,
                tools=self.tool_registry.get_openai_schema(),
            ),
        )

        return session_config.model_dump(exclude_unset=True)

    async def _send_audio_stream(self) -> None:
        """
        Sends audio data from the microphone to the OpenAI API.
        Respects the pause/resume system.
        """
        if not self.ws_manager.is_connected():
            self.logger.error("No connection available for audio transmission")
            return

        try:
            self.logger.info("Starting audio transmission...")
            audio_chunks_sent = await self._process_audio_loop()
            self.logger.info(
                "Audio transmission ended. Chunks sent: %d", audio_chunks_sent
            )
        except asyncio.TimeoutError as e:
            self.logger.error("Timeout while sending audio: %s", e)
        except Exception as e:
            self.logger.error("Error while sending audio: %s", e)

    async def _process_audio_loop(self) -> int:
        """Process the main audio streaming loop."""
        audio_chunks_sent = 0

        while self._should_continue_streaming():
            await self._wait_for_streaming_resume()

            if not self._should_continue_streaming():
                break

            if self._audio_streaming_paused:
                await asyncio.sleep(0.01)
                continue

            chunk_sent = await self._process_audio_chunk()
            if chunk_sent:
                audio_chunks_sent += 1

            await asyncio.sleep(0.01)

        return audio_chunks_sent

    def _should_continue_streaming(self) -> bool:
        """Check if audio streaming should continue."""
        return self.audio_capture.is_active and self.ws_manager.is_connected()

    async def _wait_for_streaming_resume(self) -> None:
        """Wait until audio streaming is not paused."""
        await self._audio_streaming_event.wait()

    async def _process_audio_chunk(self) -> bool:
        """Process a single audio chunk. Returns True if chunk was sent successfully."""
        data = self.audio_capture.read_chunk()
        if not data:
            await asyncio.sleep(0.01)
            return False

        success = await self.ws_manager.send_audio_stream(data)
        if not success:
            self.logger.warning("Failed to send audio chunk")

        return success

    def _register_tools(self) -> None:
        """Register available tools with the tool registry"""
        self.tool_registry.register(get_current_time)
