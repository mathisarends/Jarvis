import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


class EventBus(LoggingMixin):
    def __init__(self):
        self._subscribers: dict[VoiceAssistantEvent, list[Callable]] = {
            event_type: [] for event_type in VoiceAssistantEvent
        }
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="EventBus"
        )
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        self._subscribers[event_type] = [
            cb for cb in self._subscribers[event_type] if cb != callback
        ]

    def publish_sync(self, event_type: VoiceAssistantEvent, data: Any = None) -> None:
        loop = self._require_loop()

        for callback in self._subscribers[event_type]:
            if asyncio.iscoroutinefunction(callback):
                asyncio.run_coroutine_threadsafe(
                    self._invoke_async(callback, event_type, data), loop
                )
            else:
                loop.run_in_executor(
                    self._executor,
                    self._invoke_sync,
                    callback,
                    event_type,
                    data,
                )

    async def publish_async(
        self, event_type: VoiceAssistantEvent, data: Any = None
    ) -> None:
        for callback in self._subscribers[event_type]:
            if asyncio.iscoroutinefunction(callback):
                await self._invoke_async(callback, event_type, data)
            else:
                loop = self._require_loop()
                await loop.run_in_executor(
                    self._executor,
                    self._invoke_sync,
                    callback,
                    event_type,
                    data,
                )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _invoke_sync(
        self, callback: Callable, event: VoiceAssistantEvent, data: Any
    ) -> None:
        result = self._call_with_appropriate_args(callback, event, data)
        if asyncio.iscoroutine(result):
            self.logger.error(
                "Sync callback %s returned coroutine - use async callback",
                callback,
            )

    async def _invoke_async(
        self, callback: Callable, event: VoiceAssistantEvent, data: Any
    ) -> None:
        result = self._call_with_appropriate_args(callback, event, data)
        if asyncio.iscoroutine(result):
            await result

    def _call_with_appropriate_args(
        self, callback: Callable, event: VoiceAssistantEvent, data: Any
    ) -> Any:
        sig = inspect.signature(callback)
        params = list(sig.parameters.values())

        if params and params[0].name == "self":
            params = params[1:]

        param_count = len(params)

        if param_count == 0:
            return callback()
        elif param_count == 1:
            return callback(data) if data is not None else callback(event)
        else:
            return callback(event, data)

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError(
                "EventBus loop not attached. "
                "Call event_bus.attach_loop(asyncio.get_running_loop())"
            )
        return self._loop
