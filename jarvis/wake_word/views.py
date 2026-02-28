from enum import StrEnum


class WakeWord(StrEnum):
    ALEXA = "alexa"
    HEY_JARVIS = "hey jarvis"
    HEY_MYCROFT = "hey mycroft"
    HEY_RHASSPY = "hey rhasspy"


WAKE_WORD_MODEL: dict[WakeWord, str] = {
    WakeWord.ALEXA: "alexa_v0.1.onnx",
    WakeWord.HEY_JARVIS: "hey_jarvis_v0.1.onnx",
    WakeWord.HEY_MYCROFT: "hey_mycroft_v0.1.onnx",
    WakeWord.HEY_RHASSPY: "hey_rhasspy_v0.1.onnx",
}