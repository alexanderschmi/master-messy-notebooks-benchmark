import logging
import logging.handlers
import multiprocessing

_FORMATTER = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

def get_console_handler():
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_FORMATTER)
    return stream_handler

def get_file_handler(log_file):
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(_FORMATTER)
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
