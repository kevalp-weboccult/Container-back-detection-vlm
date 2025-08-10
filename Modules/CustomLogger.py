"""
This module provides a flexible logging class that allows individual loggers to have custom log levels.

Created by: Kaushal Shah

Usage:
    logger = CustomLogger("my_logger")
    logger.add_custom_log_level("ALERT", 55)
    logger.alert("This is an alert message.")

"""


import logging
import logging.handlers
import os
from typing import Optional, Union, Dict, Literal

class CustomLogger:
    """
    A flexible logging class that allows individual loggers to have custom log levels.
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self._custom_levels: Dict[str, int] = {}
    
    def add_custom_log_level(self, name: str, level: int) -> None:
        """
        Adds a custom logging level globally, available to all loggers.

        This ensures that the custom log method (e.g., `important`) is attached to the logging.Logger class,
        and won't raise AttributeError for any logger.
        """
        if not isinstance(name, str) or not isinstance(level, int):
            raise TypeError("Custom log level name must be a string and level must be an integer.")

        # If already defined globally, skip
        if name in logging._nameToLevel:
            return

        # If level number is already in use, raise error
        if level in logging._levelToName.values():
            raise ValueError(f"Log level '{level}' is already assigned.")

        logging.addLevelName(level, name)

        def custom_log_method(self, message: str, *args, **kwargs) -> None:
            if self.isEnabledFor(level):
                self._log(level, message, args, **kwargs)

        # Attach it to the base Logger class — not individual instances
        setattr(logging.Logger, name.lower(), custom_log_method)
    
    def get_logger(self,
        log_file: Optional[str] = "app.log",
        log_level: Union[int, str] = logging.INFO,
        log_format: Optional[str] = None,
        rotation_type: Optional[Literal["size", "time"]] = "size",
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 5,
        time_rotation_interval: Optional[Literal["S", "M", "H", "D", "midnight", "W"]] = "midnight",
        time_rotation_when: int = 1,
        log_to_console: bool = True
    ) -> logging.Logger:
        """
        Configures and returns the logger instance.

        Args:
            log_file (str, optional): The path to the log file. Default is "app.log".
            log_level (Union[int, str], optional): The log level. Default is logging.INFO.
            log_format (str, optional): The log message format. Default is None.
            rotation_type (Optional[Literal["size", "time"]], optional): The type of log rotation. Default is "size".
            max_bytes (int, optional): The maximum size of the log file in bytes. Default is 5 MB.
            backup_count (int, optional): The number of backup log files to keep. Default is 5.
            time_rotation_interval (Optional[Literal["S", "M", "H", "D", "midnight", "W"]], optional): The interval for time-based log rotation. Default is "midnight".
            time_rotation_when (int, optional): The number of times to rotate the log file. Default is 1.
            log_to_console (bool, optional): Whether to log to the console. Default is True.

        Returns:
            logging.Logger: The configured logger instance.

        """

        if not log_file:
            log_file = self.name + ".log"
        log_format = log_format or "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            if rotation_type == "size":
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=max_bytes, backupCount=backup_count
                )
            elif rotation_type == "time":
                file_handler = logging.handlers.TimedRotatingFileHandler(
                    log_file, when=time_rotation_interval, interval=time_rotation_when, backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(log_file)
            
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        return self.logger

# Usage Example
if __name__ == "__main__":
    # Logger with a custom TRACE level
    custom_logger_1 = CustomLogger("logger_one")
    custom_logger_1.add_custom_log_level("TRACE", 35)
    logger1 = custom_logger_1.get_logger("logs/logger_one.log")

    # Another logger without the TRACE level
    custom_logger_2 = CustomLogger("logger_two")
    logger2 = custom_logger_2.get_logger("logs/logger_two.log")

    logger1.info("This is an INFO message from logger one.")
    logger2.info("This is an INFO message from logger two.")
    
    logger1.trace("This is a TRACE message from logger one.")  # Works
    try:
        logger2.trace("This is a TRACE message from logger two.")  # Will raise an error
    except AttributeError:
        print("TRACE level not available for logger two.")