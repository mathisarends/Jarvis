import pyaudio
from core.config import FORMAT, CHANNELS, RATE, CHUNK
from shared.logging_mixin import LoggingMixin


class MicStream(LoggingMixin):
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_active = False
        self.audio_data = []

    def start_stream(self):
        """Start the microphone stream"""
        if self.stream is not None:
            self.stop_stream()

        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        self.is_active = True
        self.audio_data = []
        self.logger.info("Microphone stream started")

    def stop_stream(self):
        """Stop the microphone stream"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.is_active = False
        self.logger.info("Microphone stream stopped")

    def read_chunk(self):
        """Read a chunk of audio data from the microphone"""
        if self.stream and self.is_active:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            self.audio_data.append(data)
            return data
        return None

    def cleanup(self):
        """Free resources"""
        self.stop_stream()
        self.p.terminate()
