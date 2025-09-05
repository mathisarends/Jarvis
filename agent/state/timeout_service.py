import asyncio
from typing import Optional

from agent.realtime.event_bus import EventBus
from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


class TimeoutService(LoggingMixin):
    """Service for managing timeouts with EventBus integration"""

    def __init__(self, timeout_seconds: float = 20.0):
        super().__init__()
        self.timeout_seconds = timeout_seconds
        self.event_bus = EventBus()
        self._timeout_task: Optional[asyncio.Task] = None
        self._is_active = False

    async def start_timeout(self) -> None:
        """Start the timeout"""
        if self._is_active:
            self.logger.warning("Timeout already active")
            return

        self.logger.info("Starting timeout: %.1f seconds", self.timeout_seconds)
        self._is_active = True
        self._timeout_task = asyncio.create_task(self._timeout_loop())

    async def stop_timeout(self) -> None:
        """Stop the timeout"""
        if not self._is_active:
            return

        self.logger.info("Stopping timeout")
        self._is_active = False

        if not self._timeout_task or self._timeout_task.done():
            return

        self._timeout_task.cancel()
        try:
            await self._timeout_task
        except asyncio.CancelledError:  # NOSONAR
            # Intentional cleanup - CancelledError not re-raised to avoid aborting state transition
            pass
        finally:
            self._timeout_task = None

    async def _timeout_loop(self) -> None:
        """Main timeout loop"""
        try:
            await asyncio.sleep(self.timeout_seconds)
            if self._is_active:  # Only trigger if still active
                self.logger.info(
                    "Timeout occurred after %.1f seconds", self.timeout_seconds
                )
                await self._trigger_timeout()
        except asyncio.CancelledError:  # NOSONAR
            self.logger.debug("Timeout cancelled")

    async def _trigger_timeout(self) -> None:
        """Trigger timeout via EventBus"""
        try:
            self.event_bus.publish_sync(VoiceAssistantEvent.TIMEOUT_OCCURRED)
        except Exception as e:
            self.logger.exception("Error triggering timeout: %s", e)

    def update_timeout(self, new_timeout_seconds: float) -> None:
        """Update timeout duration"""
        self.timeout_seconds = new_timeout_seconds
        self.logger.info("Updated timeout to: %.1f seconds", new_timeout_seconds)

    @property
    def is_active(self) -> bool:
        """Check if timeout is currently active"""
        return self._is_active
