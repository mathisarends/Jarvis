from enum import StrEnum


class RealtimeModel(StrEnum):
    GPT_REALTIME = "gpt-realtime"
    GPT_4O_REALTIME_PREVIEW = "gpt-4o-realtime-preview"
    GPT_4O_MINI = "gpt-4o-mini"


class AssistantVoice(StrEnum):
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
