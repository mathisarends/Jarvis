import asyncio
import mimetypes
import socket
from pathlib import Path

import lameenc
import soco

from rtvoice.audio import AudioOutputDevice


class SonosAudioOutputDevice(AudioOutputDevice):
    SAMPLE_RATE = 24000
    CHANNELS = 1
    BIT_DEPTH = 16

    def __init__(self, sonos_ip: str | None = None, stream_port: int = 8765):
        self._sonos_ip = sonos_ip
        self._stream_port = stream_port
        self._speaker: soco.SoCo | None = None

        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._is_playing = False

        self._server: asyncio.Server | None = None
        self._file_server: asyncio.Server | None = None
        self._encoder = self._create_encoder()

        self._sound_files: dict[str, str] = {}
        self._sound_buffers: dict[str, bytes] = {}
        self._local_ip: str = ""

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _create_encoder(self) -> lameenc.Encoder:
        enc = lameenc.Encoder()
        enc.set_bit_rate(128)
        enc.set_in_sample_rate(self.SAMPLE_RATE)
        enc.set_channels(self.CHANNELS)
        enc.set_quality(2)
        return enc

    def _get_local_ip(self) -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()

    async def _resolve_speaker(self) -> soco.SoCo:
        if self._sonos_ip:
            return soco.SoCo(self._sonos_ip)

        devices = await asyncio.get_event_loop().run_in_executor(None, soco.discover)
        if not devices:
            raise RuntimeError("Kein Sonos-Gerät im Netzwerk gefunden.")
        return next(iter(devices))

    # -------------------------------------------------------------------------
    # MP3 Streaming Server
    # -------------------------------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break

            headers = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: audio/mpeg\r\n"
                "Cache-Control: no-cache\r\n"
                "Connection: keep-alive\r\n"
                "icy-name: Realtime Stream\r\n"
                "\r\n"
            )
            writer.write(headers.encode())
            await writer.drain()

            while self._is_playing:
                try:
                    chunk: bytes | None = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if chunk is None:
                    break

                mp3_data = self._encoder.encode(chunk)
                if mp3_data:
                    writer.write(mp3_data)
                    await writer.drain()

            try:
                final = self._encoder.flush()
                if final:
                    writer.write(final)
                    await writer.drain()
            except RuntimeError:
                pass

        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # File Server (for sound effects)
    # -------------------------------------------------------------------------

    async def _handle_file_request(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break

            parts = request_line.decode(errors="replace").split()
            filename = parts[1].lstrip("/") if len(parts) >= 2 else ""

            if filename not in self._sound_buffers:
                writer.write(b"HTTP/1.1 404 Not Found\r\n\r\n")
                await writer.drain()
                return

            data = self._sound_buffers[filename]  # direkt aus dem Buffer
            mime, _ = mimetypes.guess_type(filename)
            mime = mime or "audio/mpeg"

            headers = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: {mime}\r\n"
                f"Content-Length: {len(data)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            ).encode()
            writer.write(headers + data)
            await writer.drain()

        except Exception:
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # AudioOutputDevice interface
    # -------------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    async def start(self) -> None:
        if self._is_playing:
            return

        self._is_playing = True
        self._encoder = self._create_encoder()
        self._local_ip = self._get_local_ip()

        self._server = await asyncio.start_server(
            self._handle_client, "0.0.0.0", self._stream_port
        )
        self._file_server = await asyncio.start_server(
            self._handle_file_request, "0.0.0.0", self._stream_port + 1
        )

        stream_url = f"http://{self._local_ip}:{self._stream_port}/stream"

        self._speaker = await self._resolve_speaker()

        meta = (
            '<DIDL-Lite xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
            '<item id="1" parentID="0" restricted="1">'
            '<dc:title>Realtime Audio Stream</dc:title>'
            '<upnp:class>object.item.audioItem.musicTrack</upnp:class>'
            f'<res protocolInfo="http-get:*:audio/mpeg:*">{stream_url}</res>'
            '</item>'
            '</DIDL-Lite>'
        )

        self._speaker.play_uri(stream_url, meta=meta, title="Realtime Audio Stream")

    async def play_chunk(self, chunk: bytes) -> None:
        if self._is_playing:
            await self._queue.put(chunk)

    async def play_sound(self, file_path: str) -> None:
        if self._speaker is None:
            raise RuntimeError("SonosAudioOutputDevice nicht gestartet.")

        filename = Path(file_path).name
        self._sound_buffers[filename] = Path(file_path).read_bytes()  # direkt einlesen

        url = f"http://{self._local_ip}:{self._stream_port + 1}/{filename}"
        meta = (
            '<DIDL-Lite xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
            '<item id="1" parentID="0" restricted="1">'
            f'<dc:title>{filename}</dc:title>'
            '<upnp:class>object.item.audioItem.musicTrack</upnp:class>'
            f'<res protocolInfo="http-get:*:audio/mpeg:*">{url}</res>'
            '</item>'
            '</DIDL-Lite>'
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: self._speaker.play_uri(url, meta=meta, title=filename)
        )

    async def clear_buffer(self) -> None:
        """Discard all queued chunks and reset the MP3 encoder."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._encoder = self._create_encoder()

    async def stop(self) -> None:
        self._is_playing = False

        await self._queue.put(None)

        if self._speaker:
            try:
                self._speaker.stop()
            except Exception:
                pass

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self._file_server:
            self._file_server.close()
            await self._file_server.wait_closed()
            self._file_server = None