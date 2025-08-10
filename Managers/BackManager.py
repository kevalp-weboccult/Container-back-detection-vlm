import os 
from queue import Queue
import time
import traceback
from typing import Dict, Any, Optional, Union, List
from Modules.CustomLogger import CustomLogger
from Managers.ConfigManager import ConfigManager
from Objects.BackObject import BackObject
import cv2
from queue import Queue
from threading import Thread
from datetime import datetime, timezone

class BackManager:
    def __init__(self, name: str = "back_manager") -> None:
        """
        Initialize the BackManager with configuration settings.
        
        Args:
            name (str): Name of the back manager instance.
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
            self.logger.info(f"BackManager {self.name} initialized with settings: {self.config_manager.defaults}")
            self.all_tracked_backs: Dict[str, BackObject] = dict()
            self.current_backs: List[int] = []
            self.reader_queue: Optional[Queue[Dict[str, Any]]] = None
            self.running: bool = True
            self.missing_count: int = 10
            self.vlm_processing_queue:Queue[BackObject] = Queue(maxsize=-1)
            self.BACK_IMAGE_FOLDER = self.config_manager.get("BACK_IMAGE_FOLDER", "Processed_Images")
            if not os.path.exists(self.BACK_IMAGE_FOLDER):
                os.makedirs(self.BACK_IMAGE_FOLDER)
        except Exception as e:
            print(f"Error initializing BackManager: {e} | {traceback.format_exc()}")
    
    
    def start(self,reader_queue: Optional[Queue[Dict[str, Any]]] = None):
        """
        Start the back manager.
        """
        self.reader_queue = reader_queue
        self.logger.info(f"Starting BackManager {self.name}.")
        self.logger.info("VLM processing thread started.")
        
        while self.running:
            try:
                self.current_backs = []
                if self.reader_queue.empty():
                    self.logger.warning("Reader queue is empty. No data to process.")
                    time.sleep(1)
                    continue

                frame_data = self.reader_queue.get()
                if not frame_data:
                    self.logger.warning("Received empty frame data.")
                    continue
                bboxes = frame_data.get("detections", [])
                ids = frame_data.get("ids", [])
                frame = frame_data.get("frame", None)
                conf = frame_data.get("conf", [])
                frame_count = frame_data.get("frame_count", 0)
                inference_frame = frame.copy()

                for idx,bbox in enumerate(bboxes):
                    if ids[idx] not in self.all_tracked_backs:
                        self.all_tracked_backs[ids[idx]] = BackObject(id=ids[idx])
                        self.logger.info(f"Created new BackObject for ID {ids[idx]}.")
                    self.all_tracked_backs[ids[idx]].update(bbox, conf[idx], frame)
                
                    cv2.rectangle(inference_frame,
                                  (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[2]), int(bbox[3])),
                                  (0, 255, 0), 2)
                    cv2.putText(inference_frame, f"ID: {ids[idx]} Conf: {conf[idx]:.2f}",
                                (int(bbox[0]), int(bbox[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow("Back Tracking", inference_frame)
                cv2.waitKey(1)
                self.logger.info(f"Processed frame {frame_count} with {len(bboxes)} detections.")
                for id in list(self.all_tracked_backs.keys()):
                    if id not in ids:
                        self.all_tracked_backs[id].missing_count += 1
                        if self.all_tracked_backs[id].missing_count > self.missing_count:
                            self.logger.info(f"Removing BackObject with ID {id} as it is no longer detected.")
                            best_frame = self.all_tracked_backs[id].best_frame
                            if best_frame is not None:
                                current_time = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                                image_save_path = os.path.join(self.BACK_IMAGE_FOLDER, f"{current_time}_{id}.jpg")
                                cv2.imwrite(image_save_path, best_frame)
                            self.vlm_processing_queue.put(self.all_tracked_backs[id])
                            del self.all_tracked_backs[id]
                # You can add code to save or display the frame here if needed
                

                    


                # Initialize any necessary components or start threads here
                # For example, you might want to start a thread to monitor back objects
            except Exception as e:
                self.logger.error(f"Error starting BackManager: {e} | {traceback.format_exc()}")
    
