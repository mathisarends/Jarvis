from dataclasses import dataclass, field

from jarvis.audio import VolumeSpeakerOutput
from jarvis.events.bus import EventBus
from jarvis.tools import Timer

@dataclass
class JarvisContext:
    event_bus: EventBus
    speaker: VolumeSpeakerOutput
    timer: Timer