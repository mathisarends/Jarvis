"""
Environment variable validation for Jarvis voice assistant.
"""

import os
from dotenv import load_dotenv


def validate_environment_variables() -> None:
    """
    Validate that required environment variables are set.

    Raises:
        RuntimeError: If required environment variables are missing
    """
    # Load environment variables from .env file
    load_dotenv()

    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for real-time communication",
        "PICO_ACCESS_KEY": "Picovoice access key for wake word detection",
    }

    missing_vars = []

    for var_name, description in required_vars.items():
        if not os.getenv(var_name):
            missing_vars.append(f"{var_name} ({description})")

    if missing_vars:
        error_msg = (
            "Missing required environment variables:\n"
            + "\n".join(f"  - {var}" for var in missing_vars)
            + "\n\nPlease set these variables in your .env file or environment."
        )
        raise RuntimeError(error_msg)
