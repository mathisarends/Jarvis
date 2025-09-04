from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent.realtime.barge_in.current_message_context import CurrentMessageContext
from agent.realtime.event_types import RealtimeClientEvent
from agent.realtime.event_bus import EventBus
from agent.realtime.tools.registry import ToolRegistry
from agent.realtime.tools.tool_executor import ToolExecutor
from agent.realtime.tools.tools import get_current_time
from agent.realtime.tools.views import FunctionCallResult
from agent.realtime.transcription.service import TranscriptionService
from agent.realtime.websocket.websocket_manager import WebSocketManager
from agent.state.base import VoiceAssistantEvent
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

        # instantiate event based services
        self.current_message_timer = CurrentMessageContext()

        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.event_bus = EventBus()

        # Audio streaming control
        self._audio_streaming_paused = False
        self._audio_streaming_event = asyncio.Event()
        self._audio_streaming_event.set()  # Initially not paused

        # Subscribe to tool result events
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_RECEIVED_TOOL_CALL_RESULT,
            self._handle_tool_result,
        )

        self._register_tools()

    def pause_audio_streaming(self) -> None:
        """
        Pausiert das Audio-Streaming. Die WebSocket-Verbindung bleibt bestehen.
        """
        self.logger.info("Pausing audio streaming...")
        self._audio_streaming_paused = True
        self._audio_streaming_event.clear()

    def resume_audio_streaming(self) -> None:
        """
        Setzt das Audio-Streaming fort.
        """
        self.logger.info("Resuming audio streaming...")
        self._audio_streaming_paused = False
        self._audio_streaming_event.set()

    def is_audio_streaming_paused(self) -> bool:
        """
        Gibt zurück, ob das Audio-Streaming aktuell pausiert ist.
        """
        return self._audio_streaming_paused

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
                voice="marin",
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
                tools=self.tool_registry.get_openai_schema(),
            ),
        )

        return session_config.model_dump(exclude_unset=True)

    async def _send_audio_stream(self) -> None:
        """
        Sends audio data from the microphone to the OpenAI API.
        Respektiert das Pause/Resume-System.
        """
        if not self.ws_manager.is_connected():
            self.logger.error("No connection available for audio transmission")
            return

        try:
            self.logger.info("Starting audio transmission...")
            audio_chunks_sent = 0

            while self.audio_capture.is_active and self.ws_manager.is_connected():
                # Warten bis Audio-Streaming nicht mehr pausiert ist
                await self._audio_streaming_event.wait()

                # Nochmals prüfen, da sich der Zustand während des Wartens geändert haben könnte
                if (
                    not self.audio_capture.is_active
                    or not self.ws_manager.is_connected()
                ):
                    break

                # Wenn pausiert, kurz warten und nochmals prüfen
                if self._audio_streaming_paused:
                    await asyncio.sleep(0.01)
                    continue

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

            self.logger.info(
                "Audio transmission ended. Chunks sent: %d", audio_chunks_sent
            )

        except asyncio.TimeoutError as e:
            self.logger.error("Timeout while sending audio: %s", e)
        except Exception as e:
            self.logger.error("Error while sending audio: %s", e)

    def _register_tools(self) -> None:
        """Register available tools with the tool registry"""
        self.tool_registry.register(get_current_time)

    async def _handle_tool_result(
        self, event: VoiceAssistantEvent, data: FunctionCallResult
    ) -> None:
        """Handle tool execution results and send them back to OpenAI Realtime API"""
        try:
            self.logger.info(
                "Received tool result for '%s', sending to Realtime API", data.tool_name
            )

            # 1) Tool-Output als conversation item posten
            conversation_item = data.to_conversation_item()
            ok_item = await self.ws_manager.send_message(conversation_item)
            if not ok_item:
                self.logger.error(
                    "Failed to send function_call_output for '%s'", data.tool_name
                )
                return

            self.logger.info(
                "function_call_output for '%s' sent. Triggering response.create...",
                data.tool_name,
            )

            # 2) Modell anstoßen, das Ergebnis zu verwenden
            #    (Du kannst instructions optional leer lassen oder kurz kontextualisieren)
            response_create = {
                "type": "response.create",
            }

            ok_resp = await self.ws_manager.send_message(response_create)
            if not ok_resp:
                self.logger.error("Failed to send response.create")
                return

            self.logger.info("response.create sent successfully")

        except Exception as e:
            self.logger.error(
                "Error handling tool result for '%s': %s",
                data.tool_name,
                e,
                exc_info=True,
            )
