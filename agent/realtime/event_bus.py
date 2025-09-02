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
    Hybrid EventBus – dispatcht immer auf DEN Hauptloop.
    """
    def __init__(self):
        self._subscribers: dict[VoiceAssistantEvent, list[Callable]] = {
            event_type: [] for event_type in VoiceAssistantEvent
        }
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="EventBus")
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # später setzen!

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Den Hauptloop registrieren (einmalig beim Startup)."""
        self._loop = loop

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            # Nur für klare Fehlermeldung – NICHT automatisch erstellen!
            raise RuntimeError("EventBus loop not attached. Call event_bus.attach_loop(asyncio.get_running_loop()) during startup.")
        return self._loop

    def subscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def publish_sync(self, event_type: VoiceAssistantEvent, data: Any = None) -> None:
        """
        Aus Fremd-Threads aufrufbar. Async-Callbacks werden thread-safe
        auf den Hauptloop geschoben. Sync-Callbacks laufen im Executor.
        """
        loop = self._require_loop()
        for callback in list(self._subscribers[event_type]):
            try:
                if asyncio.iscoroutinefunction(callback):
                    fut = asyncio.run_coroutine_threadsafe(
                        self._safe_invoke_async_callback(callback, event_type, data),
                        loop,
                    )
                    # optional: Fehler loggen, ohne zu blockieren
                    fut.add_done_callback(self._callback_completed)
                else:
                    loop.run_in_executor(
                        self._executor, self._safe_invoke_sync_callback, callback, event_type, data
                    )
            except Exception as e:
                print(f"Error invoking callback for event {event_type}: {e}")

    async def publish_async(self, event_type: VoiceAssistantEvent, data: Any = None) -> None:
        """
        Aus dem Hauptloop/Async-Kontext. Async-Callbacks: await.
        Sync-Callbacks: in Executor ausführen.
        """
        loop = self._require_loop()
        for callback in list(self._subscribers[event_type]):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await self._safe_invoke_async_callback(callback, event_type, data)
                else:
                    await loop.run_in_executor(
                        self._executor, self._safe_invoke_sync_callback, callback, event_type, data
                    )
            except Exception as e:
                print(f"Error invoking async callback for event {event_type}: {e}")

    def _callback_completed(self, fut):
        try:
            fut.result()
        except Exception as e:
            print(f"Async callback failed: {e}")

    def _safe_invoke_sync_callback(self, callback: Callable, event: VoiceAssistantEvent, data: Any = None) -> None:
        sig = inspect.signature(callback)
        n = len(sig.parameters)
        if n == 0:
            callback()
        elif n == 1:
            callback(event)
        else:
            callback(event, data)

    async def _safe_invoke_async_callback(self, callback: Callable, event: VoiceAssistantEvent, data: Any = None) -> None:
        sig = inspect.signature(callback)
        n = len(sig.parameters)
        if n == 0:
            await callback()
        elif n == 1:
            await callback(event)
        else:
            await callback(event, data)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
