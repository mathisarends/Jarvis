import logging
from typing import ClassVar

from rtvoice.config import AgentEnv

LIBRARY_NAME = "rtvoice"

logger = logging.getLogger(LIBRARY_NAME)
logger.addHandler(logging.NullHandler())


def configure_logging(level: str = "WARNING") -> None:
    log_level = getattr(logging, level.upper(), logging.WARNING)

    lib_logger = logging.getLogger(LIBRARY_NAME)
    lib_logger.handlers.clear()
    lib_logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    lib_logger.addHandler(handler)


try:
    configure_logging(AgentEnv().rtvoice_log_level)
except Exception:
    configure_logging("WARNING")


class LoggingMixin:
    logger: ClassVar[logging.Logger]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.logger = logging.getLogger(f"{LIBRARY_NAME}.{cls.__name__}")
