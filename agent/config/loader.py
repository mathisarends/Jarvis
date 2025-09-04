from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from agent.config.views import VoiceAssistantConfig


def load_config_from_yaml() -> VoiceAssistantConfig:
    """
    Load and validate hierarchical YAML config from voice_assistant.yaml.

    Raises:
        FileNotFoundError: if the config file does not exist
        RuntimeError: for YAML syntax errors or validation errors
    """
    path = "voice_assistant.yaml"
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"Invalid YAML in {p}: {e}") from e

    try:
        return VoiceAssistantConfig.model_validate(raw)
    except ValidationError as e:
        raise RuntimeError(f"Invalid config values in {p}:\n{e}") from e
