import struct

from rtvoice.audio import SpeakerOutput


class VolumeSpeakerOutput(SpeakerOutput):
    def __init__(self, device_index: int | None = None, sample_rate: int = 24000):
        super().__init__(device_index=device_index, sample_rate=sample_rate)
        self._volume: float = 1.0

    @property
    def volume(self) -> int:
        return round(self._volume * 100)

    @volume.setter
    def volume(self, percent: int) -> None:
        self._volume = max(0.0, min(100.0, percent)) / 100.0

    def _apply_volume(self, chunk: bytes) -> bytes:
        if self._volume >= 1.0:
            return chunk
        if self._volume <= 0.0:
            return b"\x00" * len(chunk)
        num_samples = len(chunk) // 2
        samples = struct.unpack(f"<{num_samples}h", chunk)
        scaled = [max(-32768, min(32767, int(s * self._volume))) for s in samples]
        return struct.pack(f"<{num_samples}h", *scaled)

    def _playback_loop(self) -> None:
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            self._playing = True
            if self._stream and self._active:
                self._stream.write(self._apply_volume(chunk))
            self._playing = False