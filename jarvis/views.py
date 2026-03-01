from dataclasses import dataclass

from jarvis.audio import VolumeSpeakerOutput
from jarvis.events.bus import EventBus

@dataclass
class JarvisContext:
    event_bus: EventBus
    speaker: VolumeSpeakerOutput