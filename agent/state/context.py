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
        # Needed for initialization - prevent circular deps
        from agent.state.idle import IdleState

        self.state: AssistantState = IdleState()
        self.session_active = False
        self.wake_word_listener = wake_word_listener
        self.audio_capture = audio_capture
        self.audio_detection_service = audio_detection_service
        self.timeout_service = timeout_service
        self.event_bus = event_bus
        self.realtime_api = realtime_api

        loop = asyncio.get_running_loop()
        event_bus.attach_loop(loop)

        # Subscribe to all events and route them to handle_event
        self._setup_event_subscriptions()
        
        self.realtime_task = None
        self._realtime_session_active = False

    def _setup_event_subscriptions(self) -> None:
        """Subscribe to all VoiceAssistantEvents and route them to handle_event"""
        for event_type in VoiceAssistantEvent:
            self.event_bus.subscribe(event_type, self.handle_event)

    async def handle_event(self, event: VoiceAssistantEvent, data: Any = None) -> None:
        """Central event router - delegates events to current state"""
        await self.state.handle(event, self)

    def start_session(self) -> None:
        """Start a new session"""
        self.session_active = True

    def end_session(self) -> None:
        """End the current session"""
        self.session_active = False

    # Not sure about this here
    def start_realtime_task(self):
        if not self.realtime_task or self.realtime_task.done():
            self.realtime_task = asyncio.create_task(self.realtime_api.setup_and_run())

    async def start_realtime_session(self) -> bool:
        """
        Start a new realtime session with OpenAI.
        Returns True if started successfully, False otherwise.
        """
        if self.is_realtime_session_active():
            self.logger.warning("Realtime session already active, skipping start")
            return True

        try:
            self.logger.info("Starting realtime session...")
            self.realtime_task = asyncio.create_task(self.realtime_api.setup_and_run())
            self._realtime_session_active = True
            
            self.logger.info("Realtime session started successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to start realtime session: %s", e)
            self._realtime_session_active = False
            return False

    async def close_realtime_session(self, timeout: float = 3.0) -> bool:
        """
        Close the realtime session gracefully with timeout.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown in seconds
            
        Returns:
            True if closed successfully, False if force-cancelled
        """
        if not self.is_realtime_session_active():
            self.logger.debug("Realtime session not active, nothing to close")
            return True

        try:
            self.logger.info("Closing realtime session...")
            
            # 1. Signal graceful shutdown
            await self.realtime_api.close_connection()
            
            # 2. Wait for task to complete naturally with timeout
            if self.realtime_task and not self.realtime_task.done():
                try:
                    await asyncio.wait_for(self.realtime_task, timeout=timeout)
                    self.logger.info("Realtime session closed gracefully")
                    result = True
                    
                except asyncio.TimeoutError:
                    self.logger.warning(
                        "Realtime session didn't close within %ss, forcing cancellation", 
                        timeout
                    )
                    
                    # 3. Force cancel if timeout
                    self.realtime_task.cancel()
                    try:
                        await self.realtime_task
                    except asyncio.CancelledError:
                        self.logger.info("Realtime session force-cancelled")
                    
                    result = False
            else:
                result = True

            # 4. Reset state
            self._realtime_session_active = False
            self.realtime_task = None
            
            return result
            
        except Exception as e:
            self.logger.error("Error closing realtime session: %s", e)
            self._realtime_session_active = False
            return False

    def pause_realtime_audio(self) -> bool:
        """
        Pause the realtime audio streaming.
        WebSocket connection remains active.
        
        Returns:
            True if paused successfully, False otherwise
        """
        if not self.is_realtime_session_active():
            self.logger.warning("Cannot pause audio - realtime session not active")
            return False

        try:
            self.realtime_api.pause_audio_streaming()
            self.logger.info("Realtime audio streaming paused")
            return True
            
        except Exception as e:
            self.logger.error("Failed to pause realtime audio: %s", e)
            return False

    def resume_realtime_audio(self) -> bool:
        """
        Resume the realtime audio streaming.
        
        Returns:
            True if resumed successfully, False otherwise
        """
        if not self.is_realtime_session_active():
            self.logger.warning("Cannot resume audio - realtime session not active")
            return False

        try:
            self.realtime_api.resume_audio_streaming()
            self.logger.info("Realtime audio streaming resumed")
            return True
            
        except Exception as e:
            self.logger.error("Failed to resume realtime audio: %s", e)
            return False

    def is_realtime_audio_paused(self) -> bool:
        """
        Check if realtime audio streaming is currently paused.
        
        Returns:
            True if paused, False if active or session not running
        """
        if not self.is_realtime_session_active():
            return False
            
        return self.realtime_api.is_audio_streaming_paused()

    def is_realtime_session_active(self) -> bool:
        """
        Check if the realtime session is currently active.
        
        Returns:
            True if session is active and task is running
        """
        return (
            self._realtime_session_active 
            and self.realtime_task is not None 
            and not self.realtime_task.done()
        )