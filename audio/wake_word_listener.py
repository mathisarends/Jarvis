import asyncio
import os
import threading
from enum import Enum
from typing import Tuple, Mapping

import numpy as np
import pvporcupine
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

    def __str__(self):
        return self.value


class WakeWordListener(LoggingMixin):
    def __init__(
        self,
        wakeword: PorcupineBuiltinKeyword = PorcupineBuiltinKeyword.PICOVOICE,
        sensitivity: float = 0.7,
    ):
        if not isinstance(wakeword, PorcupineBuiltinKeyword):
            raise TypeError("wakeword must be a PorcupineBuiltinKeyword enum value")

        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError("Sensitivity must be between 0.0 and 1.0")

        self.logger.info(
            "Initializing Wake Word Listener with word: %s (Sensitivity: %.1f)",
            wakeword.value,
            sensitivity,
        )

        access_key = os.getenv("PICO_ACCESS_KEY")
        if not access_key:
            self.logger.error("PICO_ACCESS_KEY not found in environment variables")
            raise ValueError("PICO_ACCESS_KEY not found in .env file")

        self.wakeword = wakeword
        try:
            self.handle = pvporcupine.create(
                access_key=access_key,
                keywords=[wakeword.value],
                sensitivities=[sensitivity],
            )
            self.logger.info("Porcupine handle created successfully")
        except pvporcupine.PorcupineInvalidArgumentError as e:
            self.logger.error("Invalid argument for Porcupine: %s", str(e))
            raise
        except pvporcupine.PorcupineInvalidStateError as e:
            self.logger.error("Invalid state for Porcupine: %s", str(e))
            raise
        except pvporcupine.PorcupineKeyError as e:
            self.logger.error("Invalid access key for Porcupine: %s", str(e))
            raise
        except pvporcupine.PorcupineIOError as e:
            self.logger.error("IO error in Porcupine: %s", str(e))
            raise

        try:
            self.pa_input = pyaudio.PyAudio()
            self.stream = self.pa_input.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=self.handle.frame_length,
                stream_callback=self._audio_callback,
            )
            self.logger.info("Audio stream initialized successfully")
        except OSError as e:
            self.logger.error("Audio device error: %s", str(e))
            raise
        except ValueError as e:
            self.logger.error("Invalid audio parameters: %s", str(e))
            raise

        self.is_listening = False
        self.should_stop = False
        self._detection_event = threading.Event()

    def __enter__(self):
        self.logger.info("Entering WakeWordListener context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Exiting WakeWordListener context")
        self.cleanup()
        return False

    async def listen_for_wakeword_async(self) -> bool:
        """
        Asynchronous version of listen_for_wakeword that doesn't block the event loop.

        Returns:
            True if wake word was detected, False otherwise
        """
        self.logger.info("Starting async wake word listening...")
        self._detection_event.clear()
        self.is_listening = True

        if not self.stream.is_active():
            self.stream.start_stream()
            self.logger.info("Audio stream started")

        while not self.should_stop:
            detected = self._detection_event.is_set()
            if detected:
                self.logger.info("Wake word detected, returning True")
                self._detection_event.clear()
                self.is_listening = False
                return True

            await asyncio.sleep(0.1)

        self.logger.info("Wake word listening stopped")
        return False

    def stop_listening(self) -> None:
        """Stop the wake word listening"""
        self.logger.info("Stopping wake word listener")
        self.should_stop = True
        self.is_listening = False

    def cleanup(self) -> None:
        self.logger.info("Cleaning up Wake Word Listener...")
        self.should_stop = True
        self.is_listening = False

        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.logger.info("Audio stream closed")
        except OSError as e:
            self.logger.error("Error closing audio stream: %s", str(e))
        except AttributeError as e:
            self.logger.error("Stream attribute error during cleanup: %s", str(e))

        try:
            if self.pa_input:
                self.pa_input.terminate()
                self.logger.info("PyAudio terminated")
        except OSError as e:
            self.logger.error("Error terminating PyAudio: %s", str(e))

        try:
            if self.handle:
                self.handle.delete()
                self.logger.info("Porcupine handle deleted")
        except pvporcupine.PorcupineInvalidStateError as e:
            self.logger.error(
                "Invalid state when deleting Porcupine handle: %s", str(e)
            )
        except AttributeError as e:
            self.logger.error("Handle attribute error during cleanup: %s", str(e))

        self.logger.info("Wake Word Listener successfully shut down")

    def _audio_callback(
        self,
        in_data: bytes | None,
        frame_count: int,  # pylint: disable=unused-argument
        time_info: Mapping[str, float],  # pylint: disable=unused-argument
        status: int,
    ) -> Tuple[bytes | None, int]:
        """Callback for audio processing"""
        if status:
            self.logger.warning("Audio callback status: %s", status)

        if self.is_listening and not self.should_stop:
            try:
                pcm = np.frombuffer(in_data, dtype=np.int16)
                keyword_index = self.handle.process(pcm)

                if keyword_index >= 0:
                    self.logger.info("Wake word detected! Index: %d", keyword_index)
                    self._detection_event.set()
            except ValueError as e:
                self.logger.error("Data conversion error in audio callback: %s", str(e))
            except pvporcupine.PorcupineInvalidStateError as e:
                self.logger.error("Invalid Porcupine state in callback: %s", str(e))
            except pvporcupine.PorcupineInvalidArgumentError as e:
                self.logger.error(
                    "Invalid argument in Porcupine processing: %s", str(e)
                )

        return (in_data, pyaudio.paContinue)
