import os
from enum import Enum
from textwrap import dedent

from dotenv import load_dotenv

load_dotenv()


class RealtimeVoice(str, Enum):
    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"
    CEDAR = "cedar"  # only available in gpt-realtime
    MARIN = "marin"  # only available in gpt-realtime


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini-realtime-preview-2024-12-17"
OPENAI_WEBSOCKET_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}"
OPENAI_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta": "realtime=v1",
}

SYSTEM_MESSAGE = dedent(
    "Du bist ein hilfreicher KI-Assistent. Antworte klar und direkt. Wenn der Nutzer Deutsch spricht, antworte auf Deutsch."
)
VOICE = RealtimeVoice.ALLOY

TEMPERATURE = 0.8
