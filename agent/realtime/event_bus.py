from __future__ import annotations
import asyncio
import inspect
from typing import Any, Callable, Optional, get_type_hints
from concurrent.futures import ThreadPoolExecutor

from agent.state.base import VoiceAssistantEvent
from shared.logging_mixin import LoggingMixin


class EventBus(LoggingMixin):
    """
    Simple EventBus with intelligent parameter detection.
    Automatically analyzes callback signatures to determine what parameters to pass.
    """

    def __init__(self):
        self._subscribers: dict[
            VoiceAssistantEvent, list[tuple[Callable, bool, bool]]
        ] = {event_type: [] for event_type in VoiceAssistantEvent}
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="EventBus"
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the main loop (once during startup)."""
        self._loop = loop

    def subscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        """
        Subscribe to an event with automatic parameter detection based on type hints.

        Rules:
        - Parameters with type hint VoiceAssistantEvent get the event
        - Other parameters get the data
        - No parameters (after self): gets nothing
        """
        sig = inspect.signature(callback)
        try:
            type_hints = get_type_hints(callback)
        except (NameError, AttributeError):
            # Fallback if type hints can't be resolved
            type_hints = {}

        params = list(sig.parameters.keys())

        # Remove 'self' parameter if it's a method
        if params and params[0] == "self":
            params = params[1:]

        if not params:
            # No parameters -> pass nothing
            pass_event = False
            pass_data = False
        else:
            # Check type hints to determine what to pass
            pass_event = False
            pass_data = False

            for param_name in params:
                param_type = type_hints.get(param_name)
                if param_type == VoiceAssistantEvent:
                    pass_event = True
                else:
                    # Any other type (or no type hint) gets data
                    pass_data = True

        self._subscribers[event_type].append((callback, pass_event, pass_data))

    def unsubscribe(self, event_type: VoiceAssistantEvent, callback: Callable) -> None:
        """Unsubscribe a callback from an event."""
        self._subscribers[event_type] = [
            (cb, pe, pd)
            for cb, pe, pd in self._subscribers[event_type]
            if cb != callback
        ]

    def publish_sync(self, event_type: VoiceAssistantEvent, data: Any = None) -> None:
        """Publish event from any thread (thread-safe)."""
        loop = self._require_loop()

        for callback, pass_event, pass_data in self._subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    fut = asyncio.run_coroutine_threadsafe(
                        self._safe_invoke_async_callback(
                            callback, event_type, data, pass_event, pass_data
                        ),
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
                    await self._safe_invoke_async_callback(
                        callback, event_type, data, pass_event, pass_data
                    )
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
                self.logger.error(
                    f"Error invoking async callback for event {event_type}: {e}"
                )

    def _build_args(
        self, event: VoiceAssistantEvent, data: Any, pass_event: bool, pass_data: bool
    ) -> tuple:
        """Build argument tuple based on what the callback expects."""
        args = []
        if pass_event:
            args.append(event)
        if pass_data:
            args.append(data)
        return tuple(args)

    def _safe_invoke_sync_callback(
        self,
        callback: Callable,
        event: VoiceAssistantEvent,
        data: Any,
        pass_event: bool,
        pass_data: bool,
    ) -> None:
        """Safely invoke a synchronous callback with error handling."""
        try:
            args = self._build_args(event, data, pass_event, pass_data)
            callback(*args)
        except Exception as e:
            self.logger.error(f"Error in sync callback {callback}: {e}")

    async def _safe_invoke_async_callback(
        self,
        callback: Callable,
        event: VoiceAssistantEvent,
        data: Any,
        pass_event: bool,
        pass_data: bool,
    ) -> None:
        """Safely invoke an asynchronous callback with error handling."""
        try:
            args = self._build_args(event, data, pass_event, pass_data)
            await callback(*args)
        except Exception as e:
            self.logger.error(
                f"Error in async callback {callback.__name__ if hasattr(callback, '__name__') else callback}",
                exc_info=True,
            )

    def shutdown(self) -> None:
        """Shutdown the EventBus and cleanup resources."""
        self._executor.shutdown(wait=True)

    def _callback_completed(self, fut):
        """Handle completion of async callbacks scheduled from sync context."""
        try:
            fut.result()
        except Exception as e:
            self.logger.error(f"Async callback failed: {e}")

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        """Get the attached event loop or raise an error if not attached."""
        if self._loop is None:
            raise RuntimeError(
                "EventBus loop not attached. Call event_bus.attach_loop(asyncio.get_running_loop()) during startup."
            )
        return self._loop
