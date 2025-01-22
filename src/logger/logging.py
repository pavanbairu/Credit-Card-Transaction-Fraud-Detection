import os
from datetime import datetime
import logging


LOGFILE = f"log_file_{datetime.now().strftime("%m_%d_%y_%H_%M")}.log"
LOG_PATH = os.path.join(os.getcwd(), "logs", LOGFILE)

os.makedirs(LOG_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_PATH, LOGFILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
