import os 
import time
import traceback
from Modules.CustomLogger import CustomLogger
from Managers.DeviceManager import DeviceManager
from Managers.ConfigManager import ConfigManager
from typing import TYPE_CHECKING
    
class MainManager:
    def __init__(self,name:str="MainManager"):
        self.logger = CustomLogger(name)
        self.config_manager:ConfigManager = ConfigManager.get_instance()
        self.log_level = self.config_manager.defaults.get("LOG_LEVEL", "DEBUG")
        self.log_to_console = self.config_manager.defaults.get("LOG_TO_CONSOLE", True)
        self.logger = self.logger.get_logger(
            log_file=os.path.join("logs", f"{name}.log"),
            log_level=self.log_level,
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            rotation_type="size",
            max_bytes=5 * 1024 * 1024,
            backup_count=5
        )
        self.logger.info("MainManager initialized.")
        self.running:bool = True
        self.device_manager:DeviceManager = DeviceManager()
    
    def start(self):
        """
        Start the main manager.
        """
        try:
            self.logger.info("Starting MainManager.")
            self.device_manager.start()
        except Exception as e:
            self.logger.error(f"Error starting MainManager: {e} | {traceback.format_exc()}")


if __name__ == "__main__":
    main_manager = MainManager()
    main_manager.start()
    while main_manager.running:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            main_manager.running = False
            main_manager.logger.info("MainManager stopped by user.")
        
    


        
    