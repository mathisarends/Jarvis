from dataclasses import dataclass
from jarvis.events import EventBus

@dataclass
class JarvisContext:
    event_bus: EventBus