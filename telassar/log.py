import logging


def get_logger(name='telassar', level='DEBUG', stream=None,
               msg_fmt='[%(levelname)s] %(message)s', datefmt=None):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers.clear()

    stream = logging.StreamHandler(stream)
    stream.setLevel(level)
    formatter = logging.Formatter(msg_fmt, datefmt=datefmt)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
