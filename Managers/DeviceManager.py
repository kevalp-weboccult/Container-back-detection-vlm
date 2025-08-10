import os 
import traceback
from Modules.CustomLogger import CustomLogger
from Managers.ConfigManager import ConfigManager
from Managers.CameraManager import CameraManager


class DeviceManager:
    def __init__(self, name: str = "DeviceManager"):
        self.logger = CustomLogger(name)
        self.logger = self.logger.get_logger(
            log_file=os.path.join("logs", "device_manager.log"),
            log_level="DEBUG",
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            rotation_type="size",
            max_bytes=5 * 1024 * 1024,
            backup_count=5
        )
        self.logger.info("DeviceManager initialized.")
        self.running: bool = True
        self.camera_manager: CameraManager = CameraManager(name="CameraManager")


    def start(self):
        """
        Start the device manager.
        """
        try:
            self.logger.info("Starting DeviceManager.")
            self.camera_manager.start()
        except Exception as e:
            self.logger.error(f"Error starting DeviceManager: {e} | {traceback.format_exc()}")