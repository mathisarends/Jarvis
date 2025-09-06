from __future__ import annotations
import asyncio
import inspect
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor

from agent.state.base import VoiceAssistantEvent
from shared.singleton_decorator import singleton


@singleton
class EventBus:
    """
    Hybrid EventBus – always dispatches to the main loop.
    """

    def __init__(self):
        self._subscribers: dict[VoiceAssistantEvent, list[Callable]] = {
            event_type: [] for event_type in VoiceAssistantEvent
        }
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="EventBus"
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # set later!

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the main loop (once during startup)."""
        self._loop = loop

    def subscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def publish_sync(self, event_type: VoiceAssistantEvent, data: Any = None) -> None:
        """
        Callable from foreign threads. Async callbacks are thread-safely
        pushed to the main loop. Sync callbacks run in the executor.
        """
        loop = self._require_loop()
        for callback in self._subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    fut = asyncio.run_coroutine_threadsafe(
                        self._safe_invoke_async_callback(callback, event_type, data),
                        loop,
                    )
                    # optional: Log errors without blocking
                    fut.add_done_callback(self._callback_completed)
                else:
                    loop.run_in_executor(
                        self._executor,
                        self._safe_invoke_sync_callback,
                        callback,
                        event_type,
                        data,
                    )
            except Exception as e:
                print(f"Error invoking callback for event {event_type}: {e}")

    async def publish_async(
        self, event_type: VoiceAssistantEvent, data: Any = None
    ) -> None:
        """
        From the main loop/async context. Async callbacks: await.
        Sync callbacks: execute in executor.
        """
        loop = self._require_loop()
        for callback in self._subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await self._safe_invoke_async_callback(callback, event_type, data)
                else:
                    await loop.run_in_executor(
                        self._executor,
                        self._safe_invoke_sync_callback,
                        callback,
                        event_type,
                        data,
                    )
            except Exception as e:
                print(f"Error invoking async callback for event {event_type}: {e}")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _callback_completed(self, fut):
        try:
            fut.result()
        except Exception as e:
            print(f"Async callback failed: {e}")

    def _safe_invoke_sync_callback(
        self, callback: Callable, event: VoiceAssistantEvent, data: Any = None
    ) -> None:
        sig = inspect.signature(callback)
        n = len(sig.parameters)
        if n == 0:
            callback()
        elif n == 1:
            callback(event)
        else:
            callback(event, data)

    async def _safe_invoke_async_callback(
        self, callback: Callable, event: VoiceAssistantEvent, data: Any = None
    ) -> None:
        sig = inspect.signature(callback)
        print(
            f"DEBUG: {callback.__name__} has {len(sig.parameters)} parameters: {list(sig.parameters.keys())}"
        )
        n = len(sig.parameters)
        if n == 0:
            await callback()
        elif n == 1:
            await callback(event)
        else:
            await callback(event, data)

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError(
                "EventBus loop not attached. Call event_bus.attach_loop(asyncio.get_running_loop()) during startup."
            )
        return self._loop
