import os
import logging
from pathlib import Path
from datetime import datetime, date
from logging.handlers import TimedRotatingFileHandler

BASE_DIR = Path(__file__).resolve().parent
LOGS_PATH = BASE_DIR.parent / 'logs'

class ResultLogger:
    def __init__(self, log_file_name: str = "results.log", log_name: str = None):

        if log_name is None:
            log_name = f"result_logger"

        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        LOGS_PATH.mkdir(parents=True, exist_ok=True)
        # os.makedirs(LOGS_PATH, exist_ok=True)

        if self.logger.handlers:
            self.logger.handlers.clear()

        log_date = date.today()
        log_filename_parts = log_file_name.split('.', 1)

        if len(log_filename_parts) == 2:
            dated_log_filename = f"{log_filename_parts[0]}_{log_date}.{log_filename_parts[1]}"
        else:
            dated_log_filename = f"{log_file_name}_{log_date}"

        log_file_path = Path(f"{LOGS_PATH}/{dated_log_filename}")
        log_file_path.touch(exist_ok=True)

        log_handler = logging.FileHandler(log_file_path, mode='a', encoding="utf-8", delay=True)
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)
        
    def log_info(self, info_msg: str):
        self.logger.info(info_msg)
    
    def log_error(self, error_msg: str):
        self.logger.error(error_msg)


_logger_instance = None

def get_logger():
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ResultLogger()
    else:
        for handler in _logger_instance.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_path = Path(handler.baseFilename)

                if not log_path.exists():
                    for h in _logger_instance.logger.handlers[:]:
                        h.close()
                        _logger_instance.logger.removeHandler(h)
                        
                    _logger_instance = ResultLogger()
                    break
    return _logger_instance

