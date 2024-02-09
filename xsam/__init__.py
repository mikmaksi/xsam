import numpy as np
import structlog


structlog.configure(processors=[structlog.dev.ConsoleRenderer(sort_keys=False)])
logger = structlog.get_logger()

# set seed(s)
np.random.seed(1)
