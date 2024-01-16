import logging
import sys


def setup_logger(name_log):
    logger = logging.getLogger(name_log)
    logger.setLevel(logging.INFO)

    _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
    _default_handler.flush = sys.stderr.flush
    logger.addHandler(_default_handler)

    logger.propagate = False

    handlers = logger.handlers

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s] >> %(message)s")
        handler.setFormatter(formatter)

    # logger.handlers

    return logger


logger = setup_logger("sonitranslate")
logger.setLevel(logging.INFO)


def set_logging_level(verbosity_level):
    logging_level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    logger.setLevel(logging_level_mapping.get(verbosity_level, logging.INFO))
