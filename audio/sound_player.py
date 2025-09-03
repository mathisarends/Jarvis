import base64
import os
import queue
import threading
import time
import traceback
from enum import Enum
from typing import Optional

import numpy as np
import pygame
import pyaudio
from agent.realtime.event_bus import EventBus
from agent.state.base import VoiceAssistantEvent
from audio.config import AudioConfig
from shared.logging_mixin import LoggingMixin


class SoundFile(Enum):
    """Enum for available sound files."""

    ERROR = "error"
    RETURN_TO_IDLE = "return_to_idle"
    STARTUP = "startup"
    WAKE_WORD = "wake_word"


class SoundPlayer(LoggingMixin):
    """
    Unified audio player that handles both streaming audio chunks and sound file playback.
    Supports base64 encoded audio data with queuing, volume control, and sound file management.
    """

    SUPPORTED_FORMATS = {".mp3"}

    def __init__(
        self, config: Optional[AudioConfig] = None, sounds_dir: Optional[str] = None
    ):
        self.config = config or AudioConfig()

        # PyAudio setup for chunks
        self.p = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.player_thread: Optional[threading.Thread] = None
        self.current_audio_data = bytes()
        self.is_busy = False
        self.last_state_change = time.time()
        self.min_state_change_interval = 0.5
        self.stream_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.event_bus = EventBus()

        self.volume = 1.0

        # Sound file setup
        self.sounds_dir = sounds_dir or os.path.join(os.path.dirname(__file__), "res")
        self.logger.info(
            "Initializing SoundPlayer with sounds directory: %s", self.sounds_dir
        )
        self._init_mixer()

    def start_chunk_player(self):
        """Start the audio chunk player thread"""
        self.is_playing = True
        with self.stream_lock:
            self.stream = self.p.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=self.config.chunk_size,
            )
        self.player_thread = threading.Thread(target=self._play_audio_loop)
        self.player_thread.daemon = True
        self.player_thread.start()
        self.logger.info(
            "Audio chunk player started with sample rate: %d Hz",
            self.config.sample_rate,
        )

    def stop_chunk_player(self):
        """Stop the audio chunk player"""
        self.logger.info("Stopping audio chunk player")
        self.is_playing = False

        if self.player_thread:
            self.player_thread.join(timeout=2.0)

        with self.stream_lock:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

        self.p.terminate()
        self.logger.info("Audio chunk player stopped")

    def add_audio_chunk(self, base64_audio: str):
        """Add a base64 encoded audio chunk to the playback queue"""
        try:
            audio_data = base64.b64decode(base64_audio)
            self.audio_queue.put(audio_data)
            self.logger.debug(
                "Added audio chunk to queue (size: %d bytes)", len(audio_data)
            )
        except Exception as e:
            self.logger.error("Error processing audio chunk: %s", e)

    def clear_queue_and_stop_chunks(self):
        """Stop current audio playback and clear the audio queue"""
        self.logger.info("Clearing audio queue and stopping current chunk playback")

        # Clear the queue
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        # Stop and restart stream immediately
        with self.stream_lock:
            if self.stream and self.stream.is_active():
                try:
                    self.stream.stop_stream()
                    time.sleep(0.05)
                    self.stream.start_stream()
                except Exception as e:
                    self.logger.error(
                        "Error while pausing/resuming audio stream: %s", e
                    )
                    self._recreate_audio_stream()

        # Reset state with mutex protection
        with self.state_lock:
            if self.is_busy:
                self.is_busy = False
                self.current_audio_data = bytes()
                self.last_state_change = time.time()
                # Publish ASSISTANT_RESPONSE_COMPLETED when manually clearing queue
                self.event_bus.publish_sync(
                    VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED
                )

        self.logger.info("Audio queue cleared, stream kept alive")

    def is_currently_playing_chunks(self) -> bool:
        """Check if the player is currently playing audio chunks"""
        return self.is_busy

    def get_queue_size(self) -> int:
        """Get the current size of the audio queue"""
        return self.audio_queue.qsize()

    def play_sound(self, sound_name: str) -> bool:
        """Play a sound file asynchronously (non-blocking)"""
        try:
            self._validate_audio_format(sound_name)

            sound_path = self._get_sound_path(sound_name)

            if not os.path.exists(sound_path):
                self.logger.warning("Sound file not found: %s", sound_path)
                return False

            sound = pygame.mixer.Sound(sound_path)
            sound.set_volume(self.volume)
            sound.play()
            self.logger.debug("Playing sound: %s", sound_name)
            return True

        except (RuntimeError, MemoryError, UnicodeDecodeError) as e:
            self.logger.error("Error while playing %s: %s", sound_name, e)
            return False
        except ValueError as e:
            self.logger.error("Format validation failed: %s", e)
            raise
        except OSError as e:
            self.logger.error("File access error for %s: %s", sound_name, e)
            return False

    def stop_sounds(self):
        """Stop all currently playing sounds"""
        self.logger.info("Stopping all sound playback")

        # Stop pygame sounds
        if pygame.mixer.get_init():
            try:
                pygame.mixer.stop()
                self.logger.info("Pygame mixer stopped")
            except (AttributeError, RuntimeError) as e:
                self.logger.warning("Could not stop pygame mixer: %s", e)

        # Stop chunk playback
        if self.player_thread and self.player_thread.is_alive():
            with self.stream_lock:
                if self.stream:
                    try:
                        if self.stream.is_active():
                            self.stream.stop_stream()
                            self.logger.debug("Audio stream paused")
                    except Exception as e:
                        self.logger.error("Error stopping audio stream: %s", e)

        self.clear_queue_and_stop_chunks()

    def play_sound_file(self, sound_file: SoundFile) -> bool:
        """Play a sound using the SoundFile enum"""
        return self.play_sound(sound_file.value)

    def play_startup_sound(self) -> bool:
        """Play the startup sound"""
        return self.play_sound_file(SoundFile.STARTUP)

    def play_wake_word_sound(self) -> bool:
        """Play the wake word sound"""
        return self.play_sound_file(SoundFile.WAKE_WORD)

    def play_return_to_idle_sound(self) -> bool:
        """Play the return to idle sound"""
        return self.play_sound_file(SoundFile.RETURN_TO_IDLE)

    def play_error_sound(self) -> bool:
        """Play the error sound"""
        return self.play_sound_file(SoundFile.ERROR)

    def set_volume_level(self, volume: float) -> float:
        """
        Set the volume level for both chunk and file playback.
        """
        if not 0.0 <= volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0")

        self.volume = volume
        self.logger.info("Volume set to: %.2f", self.volume)
        return self.volume

    def get_volume_level(self) -> float:
        """Get the current volume level"""
        return self.volume

    def _play_audio_loop(self):
        """Thread loop for playing audio chunks"""
        while self.is_playing:
            try:
                chunk = self._get_next_audio_chunk()
                if not chunk:
                    continue

                self._process_audio_chunk(chunk)
                self.audio_queue.task_done()
                self._check_queue_state()

            except queue.Empty:
                continue
            except Exception as e:
                self._handle_playback_error(e)

    def _get_next_audio_chunk(self):
        """Get the next audio chunk from the queue"""
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def _process_audio_chunk(self, chunk):
        """Process and play an audio chunk"""
        if not chunk:
            return

        # Protect state changes with lock
        with self.state_lock:
            current_time = time.time()
            was_busy = self.is_busy
            self.is_busy = True

            # Publish ASSISTANT_STARTED_RESPONDING when we start playing after being idle
            if (
                not was_busy
                and (current_time - self.last_state_change)
                >= self.min_state_change_interval
            ):
                self.last_state_change = current_time
                # Publish event to EventBus
                self.event_bus.publish_sync(
                    VoiceAssistantEvent.ASSISTANT_STARTED_RESPONDING
                )

        adjusted_chunk = self._adjust_volume(chunk)
        self.current_audio_data = adjusted_chunk

        try:
            with self.stream_lock:
                if self.stream and self.stream.is_active():
                    self.stream.write(adjusted_chunk)
                else:
                    self.logger.warning("Stream not active, skipping chunk")
                    self._recreate_audio_stream()
        except OSError as e:
            self.logger.error("Stream write error: %s", e)
            self._recreate_audio_stream()
        except Exception as e:
            self.logger.error("Unexpected error in stream write: %s", e)
            self._recreate_audio_stream()

    def _check_queue_state(self):
        """Check the queue state and notify if playback is completed"""
        with self.state_lock:
            if self.audio_queue.empty() and self.is_busy:
                current_time = time.time()

                if (
                    current_time - self.last_state_change
                ) >= self.min_state_change_interval:
                    self.is_busy = False
                    self.current_audio_data = bytes()
                    self.last_state_change = current_time
                    # Publish ASSISTANT_RESPONSE_COMPLETED when queue is empty and we finish playing
                    self.event_bus.publish_sync(
                        VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED
                    )

    def _handle_playback_error(self, error):
        """Handle any errors during playback"""
        error_traceback = traceback.format_exc()
        self.logger.error(
            "Error playing audio chunk: %s\nTraceback:\n%s", error, error_traceback
        )

        self._recreate_audio_stream()

        with self.state_lock:
            if self.is_busy:
                self.is_busy = False
                self.last_state_change = time.time()
                # Publish ASSISTANT_RESPONSE_COMPLETED on error as well (playback stopped)
                self.event_bus.publish_sync(
                    VoiceAssistantEvent.ASSISTANT_RESPONSE_COMPLETED
                )

    def _adjust_volume(self, audio_chunk: bytes) -> bytes:
        """Adjust the volume of an audio chunk"""
        if abs(self.volume - 1.0) < 1e-6:
            return audio_chunk

        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            adjusted_array = (audio_array * self.volume).astype(np.int16)
            return adjusted_array.tobytes()
        except Exception as e:
            self.logger.error("Error adjusting volume: %s", e)
            return audio_chunk

    def _recreate_audio_stream(self):
        """Recreate the audio stream if there was an error"""
        try:
            with self.stream_lock:
                if self.stream:
                    try:
                        self.stream.close()
                    except Exception as e:
                        self.logger.warning(
                            "Error closing stream during recreation: %s", e
                        )

                try:
                    self.stream = self.p.open(
                        format=self.config.format,
                        channels=self.config.channels,
                        rate=self.config.sample_rate,
                        output=True,
                        frames_per_buffer=self.config.chunk_size,
                    )
                    self.logger.info("Audio stream recreated successfully")
                except Exception as e:
                    self.logger.error("Failed to open new stream: %s", e)
                    # Try to reinitialize PyAudio
                    self.p.terminate()
                    self.p = pyaudio.PyAudio()
                    self.stream = self.p.open(
                        format=self.config.format,
                        channels=self.config.channels,
                        rate=self.config.sample_rate,
                        output=True,
                        frames_per_buffer=self.config.chunk_size,
                    )
                    self.logger.info("PyAudio and stream recreated successfully")
        except Exception as e:
            self.logger.error("Failed to recreate audio stream: %s", e)

    def _init_mixer(self):
        """Initialize pygame mixer if not already done"""
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
                self.logger.debug("Pygame mixer initialized")
            except (RuntimeError, OSError) as e:
                self.logger.error("Failed to init pygame.mixer: %s", e)
                raise

    def _validate_audio_format(self, sound_name: str) -> None:
        """Validate that the audio format is supported"""
        if "." not in sound_name:
            return  # No extension provided, will default to .mp3

        _, ext = os.path.splitext(sound_name.lower())
        if ext in self.SUPPORTED_FORMATS:
            return  # Extension is supported

        # Unsupported extension - raise error
        supported_list = ", ".join(self.SUPPORTED_FORMATS)
        raise ValueError(
            f"Audio format '{ext}' is not supported. "
            f"Supported formats: {supported_list}. "
            f"Please convert '{sound_name}' to MP3 format."
        )

    def _get_sound_path(self, sound_name: str) -> str:
        """Get the full path to a sound file"""
        filename = sound_name if sound_name.endswith(".mp3") else f"{sound_name}.mp3"
        return os.path.join(self.sounds_dir, filename)
