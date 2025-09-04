from __future__ import annotations

import time
from typing import Optional

from agent.realtime.event_bus import EventBus
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


# Use
class CurrentMessageTimer(LoggingMixin):
    """
    Timer that tracks the duration of assistant responses.
    Returns None when starting, and duration in milliseconds when done/interrupted.

    Used for item truncation in barge-in logic when the user interrupts
    the agent while it's speaking, ensuring context remains congruent.
    """

    def __init__(self):
        self.event_bus = EventBus()
        self._start_time: Optional[float] = None

        # Subscribe to response events
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_STARTED_RESPONDING, self._on_response_started
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED, self._on_response_ended
        )
        self.event_bus.subscribe(
            VoiceAssistantEvent.ASSISTANT_SPEECH_INTERRUPTED,
            self._on_response_interrupted,
        )

        self.logger.info(
            "CurrentMessageTimer initialized and subscribed to response events"
        )

    async def _on_response_started(self, event: VoiceAssistantEvent, data=None) -> None:
        """Handle assistant started responding - start timer and return None."""
        self._start_time = time.time()
        self.logger.info("Assistant response started - timer started")
        return None

    async def _on_response_ended(
        self, event: VoiceAssistantEvent, data=None
    ) -> Optional[int]:
        """Handle assistant response completed - return duration in ms."""
        if self._start_time is None:
            self.logger.warning("Response completed but no start time recorded")
            return None

        duration_ms = int((time.time() - self._start_time) * 1000)
        self.logger.info("Assistant response completed - duration: %d ms", duration_ms)
        self._start_time = None
        return duration_ms

    async def _on_response_interrupted(
        self, event: VoiceAssistantEvent, data=None
    ) -> Optional[int]:
        """Handle assistant speech interrupted - return duration in ms."""
        if self._start_time is None:
            self.logger.warning("Response interrupted but no start time recorded")
            return None

        duration_ms = int((time.time() - self._start_time) * 1000)
        self.logger.info(
            "Assistant response interrupted - duration: %d ms", duration_ms
        )
        self._start_time = None
        return duration_ms

    def get_current_duration_ms(self) -> Optional[int]:
        """Get current duration in milliseconds if timer is running."""
        if self._start_time is None:
            return None
        return int((time.time() - self._start_time) * 1000)
