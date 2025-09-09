from __future__ import annotations

import asyncio
import base64

from agent.config.views import VoiceAssistantConfig
from agent.realtime.events.client.input_audio_buffer_append import (
    InputAudioBufferAppendEvent,
)
from agent.realtime.event_bus import EventBus
from agent.realtime.messaging.message_manager import RealtimeMessageManager
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.tool_executor import ToolExecutor
from agent.realtime.transcription.service import TranscriptionService
from agent.realtime.websocket.websocket_manager import WebSocketManager

from audio.capture import AudioCapture
from audio.player.audio_manager import AudioManager
from shared.logging_mixin import LoggingMixin


class OpenAIRealtimeAPI(LoggingMixin):

    def __init__(
        self,
        voice_assistant_config: VoiceAssistantConfig,
        audio_capture: AudioCapture,
        audio_manager: AudioManager,
        event_bus: EventBus,
    ):
        """
        Initializes the OpenAI Realtime API client.
        All configuration is loaded from configuration files.
        """
        self.audio_capture = audio_capture
        self.audio_manager = audio_manager
        self.event_bus = event_bus

        # Create WebSocketManager and TranscriptionService internally
        self.ws_manager = WebSocketManager.from_model(model=voice_assistant_config.agent.model, event_bus=event_bus)
        self.transcription_service = TranscriptionService(self.event_bus)

        self.tool_registry = ToolRegistry()

        # instantiate message manager (handles events and websocket messages)
        self.message_manager = RealtimeMessageManager(
            ws_manager=self.ws_manager,
            tool_registry=self.tool_registry,
            voice_assistant_config=voice_assistant_config,
            event_bus=self.event_bus,
        )

        self.tool_executor = ToolExecutor(
            self.tool_registry,
            self.message_manager,
            audio_manager,
            voice_assistant_config.agent,
            self.event_bus,
        )

        # Audio streaming control
        self._audio_streaming_paused = False
        self._audio_streaming_event = asyncio.Event()
        self._audio_streaming_event.set()  # Initially not paused

    async def setup_and_run(self) -> bool:
        """
        Sets up the connection and runs the main loop.
        """
        if not await self.ws_manager.create_connection():
            return False

        if not await self.message_manager.initialize_session():
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

    async def _send_audio_stream(self) -> None:
        """
        Sends audio data from the microphone to the OpenAI API.
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
        """Process the main audio streaming loop using the new async generator."""
        audio_chunks_sent = 0
        self.audio_capture.start_stream()
        try:
            async for chunk in self.audio_capture.stream_chunks():
                if not self._should_continue_streaming():
                    break
                await self._wait_for_streaming_resume()
                if self._audio_streaming_paused:
                    continue
                base64_audio_data = base64.b64encode(chunk).decode("utf-8")
                input_audio_buffer_append_event = (
                    InputAudioBufferAppendEvent.from_audio(base64_audio_data)
                )
                success = await self.ws_manager.send_message(
                    input_audio_buffer_append_event
                )
                if success:
                    audio_chunks_sent += 1
        finally:
            self.audio_capture.cleanup()
        return audio_chunks_sent

    def _should_continue_streaming(self) -> bool:
        """Check if audio streaming should continue."""
        return self.audio_capture.is_active and self.ws_manager.is_connected()

    async def _wait_for_streaming_resume(self) -> None:
        """Wait until audio streaming is not paused."""
        await self._audio_streaming_event.wait()