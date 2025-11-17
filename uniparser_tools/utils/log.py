import logging
import multiprocessing as mp
import os
from datetime import datetime
from functools import partial
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

from pytz import timezone


logger_initialized = {}
logger_with_file_initialized = {}

TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Shanghai")
FMT = "[%(asctime)s | %(process)6s] %(filename)15s:%(lineno)-5d %(levelname)8s - %(message)s"


class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return all(
            [
                record.getMessage().find("/health") == -1,
                record.getMessage().find("/version") == -1,
            ]
        )


def current_time_str():
    return datetime.now(timezone(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S.%f")


def get_root_logger(log_file=None, log_level=logging.INFO, file_mode="a"):
    """
    根 logger
    规定namespace 为 UniParser
    """
    namespace = "UniParser"
    tz = timezone(TIME_ZONE)  # UTC, Asia/Shanghai, Europe/Berlin
    return get_logger(namespace, log_file, log_level, file_mode, timezone=tz)


def get_logger(namespace="UniParser", log_file=None, log_level=logging.INFO, file_mode="a", timezone=None):
    """
    log.info(msg) or higher will print to console and file
    log.debug(msg) will only print to file
    """
    logger = logging.getLogger(namespace)
    if namespace in logger_initialized:
        prev_file_h = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        # log_file发生变更/关闭（uvicorn.run）, logger也进行变更
        if any(
            [
                log_file and not prev_file_h,  # new file handler
                log_file and len(prev_file_h) and log_file != prev_file_h[-1].baseFilename,  # file handler changed
                len(prev_file_h) and prev_file_h[-1].stream is None,  # file handler closed
            ]
        ):
            if not log_file:
                log_file = prev_file_h[-1].baseFilename  # just call same logger
            logger.manager.loggerDict.pop(namespace)
            logger_initialized.pop(namespace)
            logger_with_file_initialized.pop(namespace)

            logger = logging.getLogger(namespace)
        else:
            return logger

    logger.setLevel(logging.DEBUG)

    if timezone:
        logging.Formatter.converter = lambda *args: datetime.now(timezone).timetuple()
    logging.Formatter.default_msec_format = "%s.%03d"

    c_formatter = logging.Formatter(FMT)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(log_level)
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)

    if log_file:
        f_formatter = logging.Formatter(FMT)
        if file_mode.startswith("w"):
            f_handler = logging.FileHandler(log_file, encoding="utf-8", mode=file_mode)
        else:
            f_handler = RotatingFileHandler(log_file, encoding="utf-8", mode=file_mode, maxBytes=1024**3, backupCount=3)
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)

        logger_with_file_initialized[namespace] = True

    logger_initialized[namespace] = True

    return logger


def process_log_init(q: mp.Queue, log_file=None):
    qh = QueueHandler(q)
    qh.setLevel(logging.ERROR)  # skip this handler
    if log_file:
        logger = get_root_logger(log_file=log_file)
    else:
        logger = get_root_logger()
    # logger.handlers.clear()  # confusing ....
    logger.addHandler(qh)
    return logger


def init_multiprocess_logger(logger: logging.Logger):
    # https://stackoverflow.com/a/34964369
    log_queue = mp.Queue()
    ql = QueueListener(log_queue, *logger.handlers, respect_handler_level=True)
    ql.start()

    # enable log in process
    log_file = None
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            log_file = h.baseFilename
    init_fn = partial(process_log_init, q=log_queue, log_file=log_file)
    return init_fn
