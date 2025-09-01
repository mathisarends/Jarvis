import asyncio
from collections.abc import Generator
import os
import threading
from enum import Enum
from typing import Any, Mapping, Tuple, Optional
from contextlib import suppress, contextmanager

import numpy as np
from pvporcupine import (
    create,
    Porcupine,
    PorcupineInvalidArgumentError,
    PorcupineInvalidStateError,
    PorcupineKeyError,
    PorcupineIOError,
)
import pyaudio
from dotenv import load_dotenv

from shared.logging_mixin import LoggingMixin

load_dotenv()


class PorcupineBuiltinKeyword(Enum):
    ALEXA = "alexa"
    AMERICANO = "americano"
    BLUEBERRY = "blueberry"
    BUMBLEBEE = "bumblebee"
    COMPUTER = "computer"
    GRAPEFRUIT = "grapefruit"
    GRASSHOPPER = "grasshopper"
    HEY_GOOGLE = "hey google"
    HEY_SIRI = "hey siri"
    JARVIS = "jarvis"
    OK_GOOGLE = "ok google"
    PICOVOICE = "picovoice"
    PORCUPINE = "porcupine"
    TERMINATOR = "terminator"

    def __str__(self) -> str:
        return self.value


class WakeWordListener(LoggingMixin):
    """
    Wake word listener using Porcupine + PyAudio.

    Supports runtime reconfiguration of keyword/sensitivity by recreating the Porcupine handle.
    """

    def __init__(
        self,
        wakeword: PorcupineBuiltinKeyword = PorcupineBuiltinKeyword.PICOVOICE,
        sensitivity: float = 0.7,
    ):
        self._swap_lock = threading.Lock()
        self._detection_event = threading.Event()
        self.is_listening = False
        self.should_stop = False

        self.wakeword = self._validate_wakeword(wakeword)
        self.sensitivity = self._validate_sensitivity(sensitivity)

        self.logger.info(
            "Initializing Wake Word Listener with word=%s sensitivity=%.2f",
            self.wakeword.value,
            self.sensitivity,
        )

        self.access_key = self._load_access_key()
        self.handle = self._create_handle(self.wakeword, self.sensitivity)

        self.pa_input = pyaudio.PyAudio()
        self.stream = self._open_stream(self.handle.frame_length)

        self.logger.info("Wake Word Listener initialized")

    def __enter__(self):
        self.logger.info("Entering WakeWordListener context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Exiting WakeWordListener context")
        self.cleanup()
        return False

    async def listen_for_wakeword(self) -> bool:
        """Non-blocking wait for wake word."""
        self.logger.info("Starting async wake word listening…")
        self._detection_event.clear()
        self.is_listening = True

        if not self.stream.is_active():
            self.stream.start_stream()
            self.logger.info("Audio stream started")

        while not self.should_stop:
            if self._detection_event.is_set():
                self.logger.info("Wake word detected")
                self._detection_event.clear()
                self.is_listening = False
                return True
            await asyncio.sleep(0.1)

        self.logger.info("Wake word listening stopped")
        return False

    def stop_listening(self) -> None:
        """Stop listening loop flag."""
        self.logger.info("Stopping wake word listener")
        self.should_stop = True
        self.is_listening = False

    def set_wakeword(self, wakeword: PorcupineBuiltinKeyword) -> None:
        """Change wakeword at runtime."""
        self._reconfigure(wakeword=self._validate_wakeword(wakeword), sensitivity=None)

    def set_sensitivity(self, sensitivity: float) -> None:
        """Change sensitivity at runtime."""
        self._reconfigure(
            wakeword=None, sensitivity=self._validate_sensitivity(sensitivity)
        )

    def cleanup(self) -> None:
        """Close stream, terminate audio, delete handle."""
        self.logger.info("Cleaning up Wake Word Listener…")
        self.should_stop = True
        self.is_listening = False

        with suppress(Exception):
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.logger.info("Audio stream closed")

        with suppress(Exception):
            if self.pa_input:
                self.pa_input.terminate()
                self.logger.info("PyAudio terminated")

        with suppress(Exception):
            if self.handle:
                self.handle.delete()
                self.logger.info("Porcupine handle deleted")

        self.logger.info("Wake Word Listener shut down")

    @staticmethod
    def _validate_wakeword(
        wakeword: PorcupineBuiltinKeyword,
    ) -> PorcupineBuiltinKeyword:
        if not isinstance(wakeword, PorcupineBuiltinKeyword):
            raise TypeError("wakeword must be a PorcupineBuiltinKeyword")
        return wakeword

    @staticmethod
    def _validate_sensitivity(sens: float) -> float:
        if not 0.0 <= sens <= 1.0:
            raise ValueError("sensitivity must be between 0.0 and 1.0")
        return float(sens)

    @staticmethod
    def _load_access_key() -> str:
        key = os.getenv("PICO_ACCESS_KEY")
        if not key:
            raise ValueError("PICO_ACCESS_KEY not found in environment")
        return key

    def _create_handle(
        self,
        wakeword: PorcupineBuiltinKeyword,
        sensitivity: float,
    ) -> Porcupine:
        """Create a Porcupine handle with given config."""
        try:
            handle = create(
                access_key=self.access_key,
                keywords=[wakeword.value],
                sensitivities=[sensitivity],
            )
            self.logger.info(
                "Porcupine handle created (word=%s, sens=%.2f)",
                wakeword.value,
                sensitivity,
            )
            return handle
        except (
            PorcupineInvalidArgumentError,
            PorcupineInvalidStateError,
            PorcupineKeyError,
            PorcupineIOError,
        ) as e:
            self.logger.error("Failed to create Porcupine handle: %s", e)
            raise

    def _open_stream(self, frame_length: int):
        """Open PyAudio input stream for given frame_length."""
        try:
            stream = self.pa_input.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=frame_length,
                stream_callback=self._audio_callback,
            )
            self.logger.info("Audio stream initialized (frame_length=%d)", frame_length)
            return stream
        except (OSError, ValueError) as e:
            self.logger.error("Audio stream open failed: %s", e)
            raise

    @contextmanager
    def _paused_stream(self) -> Generator[bool, Any, None]:
        """
        Pause the current stream safely and resume after block if it was active.
        """
        was_active = False
        with suppress(Exception):
            was_active = self.stream.is_active()
        with suppress(Exception):
            if self.stream:
                self.stream.stop_stream()
        try:
            yield was_active
        finally:
            if was_active and not self.should_stop:
                with suppress(Exception):
                    self.stream.start_stream()

    def _reopen_stream_if_needed(self, new_handle: Porcupine) -> None:
        """Reopen stream if frame_length changed or stream is missing."""
        need_reopen = (
            self.handle is None
            or self.stream is None
            or new_handle.frame_length != self.handle.frame_length
        )
        if not need_reopen:
            return

        self.logger.info(
            "Reopening PyAudio stream (frame_length change or missing stream)"
        )
        with suppress(Exception):
            if self.stream:
                self.stream.close()

        self.stream = self._open_stream(new_handle.frame_length)

    def _swap_handle(self, new_handle: Porcupine) -> None:
        """Replace handle safely and delete old one."""
        old = self.handle
        self.handle = new_handle
        with suppress(Exception):
            if old:
                old.delete()

    def _reconfigure(
        self,
        wakeword: Optional[PorcupineBuiltinKeyword],
        sensitivity: Optional[float],
    ) -> None:
        """
        Swap wakeword/sensitivity at runtime by recreating the Porcupine handle.
        """
        with self._swap_lock, self._paused_stream():
            new_wake = wakeword or self.wakeword
            new_sens = sensitivity if sensitivity is not None else self.sensitivity
            self.logger.info(
                "Reconfiguring: word=%s, sensitivity=%.2f", new_wake.value, new_sens
            )

            new_handle = self._create_handle(new_wake, new_sens)
            self._reopen_stream_if_needed(new_handle)

            # Commit new config
            self._swap_handle(new_handle)
            self.wakeword = new_wake
            self.sensitivity = new_sens

            self._detection_event.clear()
            self.logger.info("Reconfiguration done.")

    def _audio_callback(
        self,
        in_data: bytes | None,
        frame_count: int,  # pylint: disable=unused-argument
        time_info: Mapping[str, float],  # pylint: disable=unused-argument
        status: int,
    ) -> Tuple[bytes | None, int]:
        """PyAudio callback: feed PCM frames into Porcupine."""
        if status:
            self.logger.warning("Audio callback status: %s", status)

        if self.is_listening and not self.should_stop and in_data:
            try:
                pcm = np.frombuffer(in_data, dtype=np.int16)
                keyword_index = self.handle.process(pcm)
                if keyword_index >= 0:
                    self.logger.info("Wake word detected (index=%d)", keyword_index)
                    self._detection_event.set()
            except (
                ValueError,
                PorcupineInvalidStateError,
                PorcupineInvalidArgumentError,
            ) as e:
                self.logger.error("Audio callback error: %s", e)

        return (in_data, pyaudio.paContinue)
