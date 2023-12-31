import logging, sys

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

    #logger.handlers

    return logger

logger = setup_logger("sonitranslate")
logger.setLevel(logging.INFO)
