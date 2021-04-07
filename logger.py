import logging
from logging.handlers import RotatingFileHandler

instance_log_file = "logs/crack.log"

logging_datefmt = "%m/%d/%Y %I:%M:%S %p"
logging_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"

logFormatter = logging.Formatter(fmt=logging_format, datefmt=logging_datefmt)

logger = logging.getLogger()
logger.setLevel(logging.NOTSET)
while (
    logger.handlers
):  # Remove un-format logging in Stream, or all of messages are appearing more than once.
    logger.handlers.pop()

if instance_log_file:
    fileHandler = RotatingFileHandler(
        filename=instance_log_file, mode="a", maxBytes=1 * 1024 * 1024, backupCount=2
    )
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

logging.getLogger("tensorflow").setLevel(logging.WARNING)
