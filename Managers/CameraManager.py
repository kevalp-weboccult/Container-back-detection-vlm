import os 
import traceback
from typing import Dict, Any, Optional, Union, List
import json
from Modules.CustomLogger import CustomLogger
from Managers.ConfigManager import ConfigManager
from Objects.CameraObject import CameraObject


class CameraManager:
    def __init__(self, name: str = "camera_manager") -> None:
        """
        Initialize the CameraManager with configuration settings.
        
        Args:
            name (str): Name of the camera manager instance.
        """
        try:
            self.name: str = name
            self.config_manager: ConfigManager = ConfigManager.get_instance()
            custom_logger: CustomLogger = CustomLogger(self.name)
            self.logger = custom_logger.get_logger(
                log_file=f"logs/{name}.log",
                log_level=self.config_manager.get("LOG_LEVEL"),
                log_to_console=self.config_manager.get("LOG_TO_CONSOLE"),
            )
            self.logger.info(f"CameraManager {self.name} initialized with settings: {self.config_manager.defaults}")
            
            self.url: Union[str,int] = self.config_manager.get("CAMERA_URL","20250731_173123.mp4")
            self.camera_object = CameraObject(name=self.name, url=self.url, camera_manger=self)
            

            
        except Exception as e:
            print(f"Error initializing CameraManager: {e} | {traceback.format_exc()}")
    
    def start(self):
        """
        Start the camera manager.
        """
        try:
            self.logger.info(f"Starting CameraManager {self.name} with URL: {self.url}")
            self.camera_object.start()
        except Exception as e:
            self.logger.error(f"Error starting CameraManager: {e} | {traceback.format_exc()}")