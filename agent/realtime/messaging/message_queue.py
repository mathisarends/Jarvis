import asyncio
from collections import deque
from collections.abc import Callable

from shared.logging_mixin import LoggingMixin


class MessageQueue(LoggingMixin):
    def __init__(self):
        self._response_active = False
        self._message_queue = deque()
        self._queue_processing = False

    async def send_or_queue(self, message_func: Callable, *args, **kwargs) -> None:
        if self._response_active:
            self.logger.debug("Response active - queueing message")
            self._message_queue.append((message_func, args, kwargs))
        else:
            self.logger.debug("No active response - sending message immediately")
            await message_func(*args, **kwargs)

    async def process_queue(self) -> None:
        if self._queue_processing:
            return

        self._queue_processing = True
        try:
            while self._message_queue and not self._response_active:
                message_func, args, kwargs = self._message_queue.popleft()
                self.logger.debug("Processing queued message")
                await message_func(*args, **kwargs)
                await asyncio.sleep(0.1)  # Rate limiting
        except Exception as e:
            self.logger.error("Error processing message queue: %s", e, exc_info=True)
        finally:
            self._queue_processing = False

    def set_response_active(self, active: bool) -> None:
        self._response_active = active
