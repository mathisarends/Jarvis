"""
VoiceAssistantContext - Central context holding all services and state management.
Provides access to audio services, timeout management, and wake word detection.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from agent.realtime.event_bus import EventBus
from agent.realtime.realtime_api import OpenAIRealtimeAPI
from audio.capture import AudioCapture
from audio.detection import AudioDetectionService
from agent.state.timeout_service import TimeoutService
from shared.logging_mixin import LoggingMixin
from agent.state.base import VoiceAssistantEvent


if TYPE_CHECKING:
    from audio.wake_word_listener import WakeWordListener
    from agent.state.base import AssistantState


class VoiceAssistantContext(LoggingMixin):
    """Context object that holds state and dependencies"""

    def __init__(
        self,
        wake_word_listener: WakeWordListener,
        audio_capture: AudioCapture,
        audio_detection_service: AudioDetectionService,
        timeout_service: TimeoutService,
        event_bus: EventBus,
        realtime_api: OpenAIRealtimeAPI,
    ):
        from agent.state.idle import IdleState

        self.state: AssistantState = IdleState()
        self.wake_word_listener = wake_word_listener
        self.audio_capture = audio_capture
        self.audio_detection_service = audio_detection_service
        self.timeout_service = timeout_service
        self.event_bus = event_bus
        self.realtime_api = realtime_api

        loop = asyncio.get_running_loop()
        event_bus.attach_loop(loop)
        self._setup_event_subscriptions()

        self.realtime_task = None

    def _setup_event_subscriptions(self) -> None:
        """Subscribe to all VoiceAssistantEvents and route them to handle_event"""
        for event_type in VoiceAssistantEvent:
            self.event_bus.subscribe(event_type, self.handle_event)

    async def handle_event(self, event: VoiceAssistantEvent, data: Any = None) -> None:
        """Central event router - delegates events to current state"""
        await self.state.handle(event, self)

    async def start_realtime_session(self) -> bool:
        """Start a new realtime session with OpenAI, returns success status"""
        if self._is_realtime_session_active():
            self.logger.warning("Realtime session already active, skipping start")
            return True

        try:
            self.logger.info("Starting realtime session...")
            self.realtime_task = asyncio.create_task(self.realtime_api.setup_and_run())
            self.logger.info("Realtime session started successfully")
            return True
        except Exception as e:
            self.logger.error("Failed to start realtime session: %s", e)
            return False

    async def close_realtime_session(self, timeout: float = 1.0) -> bool:
        """Close realtime session, should complete quickly after WebSocket close"""
        if not self._is_realtime_session_active():
            return True

        try:
            await self.realtime_api.close_connection()

            # Task sollte jetzt schnell beenden
            if self.realtime_task:
                await asyncio.wait_for(self.realtime_task, timeout=timeout)

            self.realtime_task = None
            return True

        except asyncio.TimeoutError:
            self.logger.error("Task didn't complete - this should not happen!")
            self.realtime_task.cancel()
            return False

    def ensure_realtime_audio_channel_paused(self) -> None:
        """Ensures realtime audio channel is paused, throws RuntimeError if session inactive"""
        if not self._is_realtime_session_active():
            raise RuntimeError("Cannot pause audio - realtime session not active")

        if not self._is_realtime_audio_paused():
            self.realtime_api.pause_audio_streaming()
            self.logger.info("Realtime audio streaming paused")

    async def ensure_realtime_audio_channel_connected(self) -> None:
        """Ensures realtime audio channel is connected, starts session if not active"""
        if not self._is_realtime_session_active():
            self.logger.info("Realtime session not active, starting new session...")
            success = await self.start_realtime_session()
            if not success:
                raise RuntimeError("Failed to start realtime session")

        if not self.audio_capture.is_active:
            self.audio_capture.start_stream()
            self.logger.info("Microphone stream reactivated")

        if self._is_realtime_audio_paused():
            self.realtime_api.resume_audio_streaming()
            self.logger.info("Realtime audio streaming resumed")

    def _is_realtime_audio_paused(self) -> bool:
        """Check if realtime audio streaming is currently paused"""
        if not self._is_realtime_session_active():
            return False
        return self.realtime_api.is_audio_streaming_paused()

    def _is_realtime_session_active(self) -> bool:
        """Check if the realtime session is currently active"""
        return self.realtime_task is not None and not self.realtime_task.done()
