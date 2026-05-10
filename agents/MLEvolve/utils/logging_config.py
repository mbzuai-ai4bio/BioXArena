import logging
import sys
from typing import Any


class VerboseFilter(logging.Filter):
    """Filter out records marked with the verbose attribute."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (hasattr(record, "verbose") and record.verbose)


def setup_logging(cfg: Any) -> logging.Logger:
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format=log_format,
        handlers=[],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger("MLEvolve")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(cfg.log_dir / "MLEvolve.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "MLEvolve.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)
    return logger
