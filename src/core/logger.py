import logging
import logging.handlers
import multiprocessing

_LEVEL_COLORS = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"

_FILE_FORMATTER = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        color = _LEVEL_COLORS.get(record.levelno, "")
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = f"{color}{record.levelname}{_RESET}"
        record.msg = f"{color}{record.msg}{_RESET}"
        return super().format(record)


_COLOR_FORMATTER = _ColorFormatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def get_console_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_COLOR_FORMATTER)
    return stream_handler

def get_file_handler(log_file):
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(_FILE_FORMATTER)
    return file_handler

def setup_logger(log_file=None, use_multiprocessing=False):
    """
    Configure the root logger.
    If use_multiprocessing is True, returns (log_queue, queue_listener) 
    which the caller must start() and stop().
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    handlers = [get_console_handler()]
    if log_file:
        handlers.append(get_file_handler(log_file))

    if use_multiprocessing:
        m = multiprocessing.Manager()
        log_queue = m.Queue()
        queue_listener = logging.handlers.QueueListener(log_queue, *handlers)
        
        # When multiprocessing, the root logger in the main process should also just use the handlers
        # or we could make the main process just write to the queue, but writing to handlers directly 
        # is fine for the main process.
        for h in handlers:
            root_logger.addHandler(h)

        return log_queue, queue_listener
    else:
        for h in handlers:
            root_logger.addHandler(h)
        return None, None

def worker_init(q):
    """Initialize logging in worker processes to send logs to the central queue."""
    qh = logging.handlers.QueueHandler(q)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(qh)
