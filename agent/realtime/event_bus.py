from __future__ import annotations
import asyncio
import inspect
from typing import Any, Callable, Optional, TypeVar
from concurrent.futures import ThreadPoolExecutor

from agent.state.base import VoiceAssistantEvent

F = TypeVar('F', bound=Callable)


class EventBus:
    """
    EventBus with instance-based decorator support.
    Use @event_bus.on_event() directly instead of register_handlers().
    """

    def __init__(self):
        self._subscribers: dict[VoiceAssistantEvent, list[tuple[Callable, bool, bool]]] = {
            event_type: [] for event_type in VoiceAssistantEvent
        }
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="EventBus"
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the main loop (once during startup)."""
        self._loop = loop

    def subscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        """
        Manual subscription (backward compatibility).
        Prefer using @event_bus.on_event() decorators instead.
        """
        # Check if callback has event handler metadata from old decorators
        pass_event = getattr(callback, '_pass_event', True)  
        pass_data = getattr(callback, '_pass_data', True)   
        
        self._subscribers[event_type].append((callback, pass_event, pass_data))

    def register_handlers(self, handler_object: Any) -> None:
        """
        Auto-register decorated methods (backward compatibility).
        Only needed for old-style @on_event decorators.
        """
        for attr_name in dir(handler_object):
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
            
            attr = getattr(handler_object, attr_name)
            
            if hasattr(attr, '_event_type'):
                event_type = attr._event_type
                pass_event = attr._pass_event
                pass_data = attr._pass_data
                
                self._subscribers[event_type].append((attr, pass_event, pass_data))
                print(f"Registered event handler: {attr_name} for {event_type}")

    def unsubscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        """Unsubscribe a callback from an event."""
        self._subscribers[event_type] = [
            (cb, pe, pd) for cb, pe, pd in self._subscribers[event_type] 
            if cb != callback
        ]

    def publish_sync(self, event_type: VoiceAssistantEvent, data: Any = None) -> None:
        """Publish event from any thread (thread-safe)."""
        loop = self._require_loop()
        
        for callback, pass_event, pass_data in self._subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    fut = asyncio.run_coroutine_threadsafe(
                        self._safe_invoke_async_callback(callback, event_type, data, pass_event, pass_data),
                        loop,
                    )
                    fut.add_done_callback(self._callback_completed)
                else:
                    loop.run_in_executor(
                        self._executor,
                        self._safe_invoke_sync_callback,
                        callback,
                        event_type,
                        data,
                        pass_event,
                        pass_data,
                    )
            except Exception as e:
                print(f"Error invoking callback for event {event_type}: {e}")

    async def publish_async(
        self, event_type: VoiceAssistantEvent, data: Any = None
    ) -> None:
        """Publish event from async context."""
        for callback, pass_event, pass_data in self._subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await self._safe_invoke_async_callback(callback, event_type, data, pass_event, pass_data)
                else:
                    loop = self._require_loop()
                    await loop.run_in_executor(
                        self._executor,
                        self._safe_invoke_sync_callback,
                        callback,
                        event_type,
                        data,
                        pass_event,
                        pass_data,
                    )
            except Exception as e:
                print(f"Error invoking async callback for event {event_type}: {e}")

    def _build_args(self, event: VoiceAssistantEvent, data: Any, pass_event: bool, pass_data: bool) -> tuple:
        """Build argument tuple based on what the callback expects."""
        args = []
        if pass_event:
            args.append(event)
        if pass_data:
            args.append(data)
        return tuple(args)

    def _safe_invoke_sync_callback(
        self, callback: Callable, event: VoiceAssistantEvent, data: Any, 
        pass_event: bool, pass_data: bool
    ) -> None:
        """Safely invoke a synchronous callback with error handling."""
        try:
            args = self._build_args(event, data, pass_event, pass_data)
            callback(*args)
        except Exception as e:
            print(f"Error in sync callback {callback}: {e}")

    async def _safe_invoke_async_callback(
        self, callback: Callable, event: VoiceAssistantEvent, data: Any,
        pass_event: bool, pass_data: bool
    ) -> None:
        """Safely invoke an asynchronous callback with error handling."""
        try:
            args = self._build_args(event, data, pass_event, pass_data)
            await callback(*args)
        except Exception as e:
            print(f"Error in async callback {callback}: {e}")

    def shutdown(self) -> None:
        """Shutdown the EventBus and cleanup resources."""
        self._executor.shutdown(wait=True)

    def _callback_completed(self, fut):
        """Handle completion of async callbacks scheduled from sync context."""
        try:
            fut.result()
        except Exception as e:
            print(f"Async callback failed: {e}")

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        """Get the attached event loop or raise an error if not attached."""
        if self._loop is None:
            raise RuntimeError(
                "EventBus loop not attached. Call event_bus.attach_loop(asyncio.get_running_loop()) during startup."
            )
        return self._loop

def on_event(event_type: VoiceAssistantEvent) -> Callable[[F], F]:
    """Standalone decorator for handlers that don't need event or data parameters."""
    def decorator(func: F) -> F:
        func._event_type = event_type
        func._pass_event = False
        func._pass_data = False
        return func
    return decorator


def on_event_with_data(event_type: VoiceAssistantEvent) -> Callable[[F], F]:
    """Standalone decorator for handlers that only need the data parameter."""
    def decorator(func: F) -> F:
        func._event_type = event_type
        func._pass_event = False
        func._pass_data = True
        return func
    return decorator


def on_event_full(event_type: VoiceAssistantEvent) -> Callable[[F], F]:
    """Standalone decorator for handlers that need both event and data parameters."""
    def decorator(func: F) -> F:
        func._event_type = event_type
        func._pass_event = True
        func._pass_data = True
        return func
    return decorator