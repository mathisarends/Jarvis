from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from agent.realtime.views import AssistantVoice
from audio.wake_word_listener import PorcupineBuiltinKeyword


class TurnDetectionConfig(BaseModel):
    """Configuration for turn detection"""

    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    type: str = "server_vad"


class AgentConfig(BaseModel):
    """Configuration for the agent"""

    voice: AssistantVoice = AssistantVoice.MARIN
    instructions: Optional[str] = None
    temperature: float = 0.8
    speed: float = 1.0
    turn_detection: Optional[TurnDetectionConfig] = None

    @field_validator("voice", mode="before")
    @classmethod
    def _coerce_voice(cls, v: Any) -> Any:
        if isinstance(v, AssistantVoice) or v is None:
            return v
        if isinstance(v, str):
            name = v.strip().upper()
            try:
                return AssistantVoice[name]
            except Exception:
                for member in AssistantVoice:
                    if member.value.lower() == v.strip().lower():
                        return member
        raise ValueError(f"Invalid voice: {v!r}")


class WakeWordConfig(BaseModel):
    """Configuration for the wake word"""

    keyword: PorcupineBuiltinKeyword = PorcupineBuiltinKeyword.PICOVOICE
    sensitivity: float = Field(0.7, ge=0.0, le=1.0)

    @field_validator("keyword", mode="before")
    @classmethod
    def _coerce_keyword(cls, v: Any) -> Any:
        if isinstance(v, PorcupineBuiltinKeyword) or v is None:
            return v
        if isinstance(v, str):
            name = v.strip().upper()
            try:
                return PorcupineBuiltinKeyword[name]
            except Exception:
                for member in PorcupineBuiltinKeyword:
                    if member.value.lower() == v.strip().lower():
                        return member
        raise ValueError(f"Invalid wake_word: {v!r}")


class AppConfig(BaseModel):
    """Main application configuration"""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)


def load_config(path: str | Path) -> AppConfig:
    """
    Load and validate hierarchical YAML config.

    Raises:
        FileNotFoundError: if the file does not exist
        RuntimeError: for YAML syntax errors or validation errors
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"Invalid YAML in {p}: {e}") from e

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        raise RuntimeError(f"Invalid config values in {p}:\n{e}") from e


# Test CLI
if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "voice_assistant.yaml"
    cfg = load_config(cfg_path)
    print(cfg.model_dump())
