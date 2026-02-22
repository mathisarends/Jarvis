import os
from dotenv import load_dotenv

load_dotenv(override=True)

def configure_logging():
    import logging
    import sys

    raw_level = os.getenv("JARVIS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, raw_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )