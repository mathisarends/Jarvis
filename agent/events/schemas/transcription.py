from enum import StrEnum
from typing import Any

from pydantic import BaseModel, field_validator


class TranscriptionModel(StrEnum):
    WHISPER_1 = "whisper-1"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"


class InputAudioTranscriptionConfig(BaseModel):
    model: TranscriptionModel = TranscriptionModel.WHISPER_1
    language: str | None = None
    prompt: str | None = None

    @field_validator("language", mode="before")
    @classmethod
    def validate_language(cls, v: Any) -> str | None:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            lang = v.strip().lower()
            if len(lang) in (2, 3) and lang.isalpha():
                return lang
        raise ValueError(
            f"Invalid language code: {v!r}. Expected ISO-639-1 format (e.g., 'en', 'de')"
        )
