"""
Configuration Manager Module.

This module provides a singleton ConfigManager class for handling application configuration.
It manages default settings, private settings, and configuration persistence through JSON files.
The ConfigManager follows the singleton pattern to ensure only one instance exists throughout
the application lifecycle.

Classes:
    ConfigManager: Singleton class for managing application configuration settings.

Example:
    >>> config = ConfigManager.get_instance()
    >>> config.set("LOG_LEVEL", 20)
    >>> log_level = config.get("LOG_LEVEL", 10)
"""

import json
import os
import sys
from typing import Any, Dict, Optional, Union, ClassVar


class ConfigManager:
    """
    Singleton Configuration Manager for application settings.
    
    This class manages application configuration including default settings,
    private settings, and configuration file persistence. It implements the
    singleton pattern to ensure only one instance exists.
    
    Attributes:
        _instance (Optional[ConfigManager]): Class variable holding the singleton instance.
        defaults (Dict[str, Any]): Dictionary containing default configuration values.
        private (Dict[str, Optional[str]]): Dictionary containing private/sensitive settings.
    """
    
    _instance: ClassVar[Optional['ConfigManager']] = None

    @staticmethod
    def get_instance() -> 'ConfigManager':
        """
        Get the singleton instance of ConfigManager.
        
        Creates a new instance if one doesn't exist, otherwise returns the existing instance.
        
        Returns:
            ConfigManager: The singleton instance of ConfigManager.
        """
        if ConfigManager._instance is None:
            ConfigManager._instance = ConfigManager()
        return ConfigManager._instance
    
    def __init__(self) -> None:
        """
        Initialize the ConfigManager instance.
        
        This constructor enforces the singleton pattern by raising an exception
        if an instance already exists. Sets up default and private configuration
        dictionaries and loads configuration from file.
        
        Raises:
            Exception: If an instance of ConfigManager already exists.
        """
        if ConfigManager._instance is not None:
            raise Exception("ConfigManager is a singleton!")
        else:
            ConfigManager._instance = self

        self.defaults: Dict[str, Any] = {
            "LOG_LEVEL": 10,
            "LOG_TO_CONSOLE": True,
            "LOG_BACKUP_COUNT": 5,
            'MODEL_IMGSZ': [640, 640],
            'USE_YOLO': False,
            'CONFIDENCE': 0.5,
            'IOU': 0.45,
            'IS_LIVE': False,
            'YOLO_MODEL_PATH':"yolov8n.pt",
            "CAMERA_URLS": [],
            "CAMERA_URL":"20250731_173123.mp4"
        }
        
        self.private: Dict[str, Optional[str]] = {
            "AWS_ACCESS_KEY_ID": None,
            "AWS_SECRET_ACCESS_KEY": None,
            "AWS_DEFAULT_REGION": None,
            "AWS_BUCKET": None,
            "AWS_URL": None,
            "REMOTE_DB_URI": None,
        }
        
        self.read_config_file()
    

    def read_config_file(self) -> None:
        """
        Read configuration from config.json file.
        
        If the config.json file exists, loads the configuration and updates the defaults.
        If the file doesn't exist, creates a new config file with current defaults.
        
        Raises:
            json.JSONDecodeError: If the config file contains invalid JSON.
            IOError: If there's an error reading the config file.
        """
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r", encoding="utf-8") as config_file:
                    config: Dict[str, Any] = json.load(config_file)
                    self.defaults.update(config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading config file: {e}")
                self.save_config_file()
        else:
            self.save_config_file()

    def save_config_file(self) -> None:
        """
        Save current configuration to config.json file.
        
        Writes the current defaults dictionary to config.json with proper indentation.
        
        Raises:
            IOError: If there's an error writing to the config file.
        """
        try:
            with open("config.json", "w", encoding="utf-8") as config_file:
                json.dump(self.defaults, config_file, indent=4)
        except IOError as e:
            print(f"Error saving config file: {e}")

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value and save to file.
        
        Args:
            key (str): The configuration key to set.
            value (Any): The value to assign to the key.
        """
        self.defaults[key] = value
        self.save_config_file()
        
    def set_all(self, settings: Dict[str, Any]) -> None:
        """
        Update multiple configuration values and save to file.
        
        Args:
            settings (Dict[str, Any]): Dictionary of key-value pairs to update.
        """
        self.defaults.update(settings)
        self.save_config_file()

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value from defaults or private settings.
        
        Args:
            key (str): The configuration key to retrieve.
            default (Optional[Any]): Default value to return if key is not found.
            
        Returns:
            Any: The configuration value, or default if not found.
        """
        if key in self.defaults:
            return self.defaults[key]
        elif key in self.private:
            return self.private[key]
        else:
            return default
        
    def get_private(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a private configuration value.
        
        Args:
            key (str): The private configuration key to retrieve.
            default (Optional[str]): Default value to return if key is not found.
            
        Returns:
            Optional[str]: The private configuration value, or default if not found.
        """
        if key in self.private:
            return self.private[key]
        else:
            return default

    def set_private(self, key: str, value: Optional[str]) -> None:
        """
        Set a private configuration value.
        
        Args:
            key (str): The private configuration key to set.
            value (Optional[str]): The value to assign to the private key.
        """
        self.private[key] = value
    

