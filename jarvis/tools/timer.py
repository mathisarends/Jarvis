import asyncio
from dataclasses import dataclass
from enum import StrEnum

from importlib.resources import files
import sounddevice as sd
import soundfile as sf


class TimerState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    RINGING = "ringing"


@dataclass
class ActionResult:
    success: bool
    message: str


@dataclass
class TimerStatus:
    state: TimerState
    remaining_seconds: int | None = None
    duration_seconds: int | None = None


class Timer:
    _TIMER_SOUND = str(files("jarvis.sounds").joinpath("timer_ringing.mp3"))

    def __init__(self) -> None:
        self._state = TimerState.IDLE
        self._task: asyncio.Task | None = None
        self._remaining: int | None = None
        self._duration: int | None = None

        self._ring_data, self._samplerate = sf.read(self._TIMER_SOUND, dtype="float32")
        self._ring_task: asyncio.Task | None = None

    def start(self, seconds: int) -> ActionResult:
        if self._state == TimerState.RUNNING:
            return ActionResult(success=False, message="A timer is already running. Please delete it first.")

        self._duration = seconds
        self._state = TimerState.RUNNING
        self._task = asyncio.create_task(self._run(seconds))

        minutes, secs = divmod(seconds, 60)
        if minutes:
            label = f"{minutes} minute{'s' if minutes > 1 else ''}" + (f" and {secs} seconds" if secs else "")
        else:
            label = f"{seconds} seconds"

        return ActionResult(success=True, message=f"Timer started for {label}.")

    def stop_ringing(self) -> ActionResult:
        if self._state != TimerState.RINGING:
            return ActionResult(success=False, message="No timer is currently ringing.")

        sd.stop()
        if self._ring_task:
            self._ring_task.cancel()
        self._state = TimerState.IDLE
        return ActionResult(success=True, message="Timer stopped.")

    def delete(self) -> ActionResult:
        if self._state == TimerState.IDLE:
            return ActionResult(success=False, message="No active timer to delete.")

        if self._task:
            self._task.cancel()
        if self._ring_task:
            self._ring_task.cancel()

        sd.stop()
        self._state = TimerState.IDLE
        self._duration = None
        self._remaining = None
        return ActionResult(success=True, message="Timer deleted.")

    def status(self) -> TimerStatus:
        return TimerStatus(
            state=self._state,
            remaining_seconds=self._remaining,
            duration_seconds=self._duration,
        )

    async def _run(self, seconds: int) -> None:
        self._remaining = seconds
        try:
            while self._remaining > 0:
                await asyncio.sleep(1)
                self._remaining -= 1

            self._state = TimerState.RINGING
            self._ring_task = asyncio.create_task(self._ring())
        except asyncio.CancelledError:
            pass

    async def _ring(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while self._state == TimerState.RINGING:
                await loop.run_in_executor(
                    None,
                    lambda: sd.play(self._ring_data, self._samplerate, blocking=True),
                )
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            sd.stop()