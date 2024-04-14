import logging
import sys
import warnings
import os


def configure_logging_libs(debug=False):
    warnings.filterwarnings(
      action="ignore", category=UserWarning, module="pyannote"
    )
    modules = [
      "numba", "httpx", "markdown_it", "speechbrain", "fairseq", "pyannote",
      "faiss",
      "pytorch_lightning.utilities.migration.utils",
      "pytorch_lightning.utilities.migration",
      "pytorch_lightning",
      "lightning",
      "lightning.pytorch.utilities.migration.utils",
    ]
    try:
        for module in modules:
            logging.getLogger(module).setLevel(logging.WARNING)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3" if not debug else "1"

        # fix verbose pyannote audio
        def fix_verbose_pyannote(*args, what=""):
            pass
        import pyannote.audio.core.model # noqa
        pyannote.audio.core.model.check_version = fix_verbose_pyannote
    except Exception as error:
        logger.error(str(error))


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
