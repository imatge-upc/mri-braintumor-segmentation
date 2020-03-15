import logging

# create logger
logger = logging.getLogger('brain_logger')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(level=logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s %(levelname)s - [%(module)s/%(funcName)s] - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)