import asyncio
from importlib.resources import files

from soco import discover

from jarvis.audio import SonosAudioOutputDevice

_WAKE_SOUND = str(files("jarvis.sounds").joinpath("wakesound.mp3"))
_STOPPED_SOUND = str(files("jarvis.sounds").joinpath("agent_stopped.mp3"))
_ERROR_SOUND = str(files("jarvis.sounds").joinpath("error_sound.mp3"))


async def main() -> None:
    device = SonosAudioOutputDevice(sonos_ip="192.168.178.68")
    await device.start()

    for name, path in [
        ("Wake Sound", _WAKE_SOUND),
        ("Stopped Sound", _STOPPED_SOUND),
        ("Error Sound", _ERROR_SOUND),
    ]:
        print(f"\nPlaying {name}...")
        await device.play_sound(path)
        await asyncio.sleep(4)

    await device.stop()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())