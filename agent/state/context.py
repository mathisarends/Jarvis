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
from audio.sound_player import SoundPlayer
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