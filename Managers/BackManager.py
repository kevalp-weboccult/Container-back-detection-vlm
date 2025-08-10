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
            self.Capture_save_folder = self.config_manager.get("Capture_save_folder", "Processed_Images")
            if not os.path.exists(self.Capture_save_folder):
                os.makedirs(self.Capture_save_folder)
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

                for idx,bbox in enumerate(bboxes):
                    if ids[idx] not in self.all_tracked_backs:
                        self.all_tracked_backs[ids[idx]] = BackObject(id=ids[idx])
                        self.logger.info(f"Created new BackObject for ID {ids[idx]}.")
                    self.all_tracked_backs[ids[idx]].update(bbox, conf[idx], frame)
                
                    cv2.rectangle(frame,
                                  (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[2]), int(bbox[3])),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {ids[idx]} Conf: {conf[idx]:.2f}",
                                (int(bbox[0]), int(bbox[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow("Back Tracking", frame)
                cv2.waitKey(1)
                self.logger.info(f"Processed frame {frame_count} with {len(bboxes)} detections.")
                for id in list(self.all_tracked_backs.keys()):
                    if id not in ids:
                        self.all_tracked_backs[id].missing_count += 1
                        if self.all_tracked_backs[id].missing_count > self.missing_count:
                            self.logger.info(f"Removing BackObject with ID {id} as it is no longer detected.")
                            self.vlm_processing_queue.put(self.all_tracked_backs[id])
                            del self.all_tracked_backs[id]
                # You can add code to save or display the frame here if needed
                

                    


                # Initialize any necessary components or start threads here
                # For example, you might want to start a thread to monitor back objects
            except Exception as e:
                self.logger.error(f"Error starting BackManager: {e} | {traceback.format_exc()}")
    
    def vlm_processing(self):
        """
        Thread to process BackObjects for VLM (Vision Language Model) processing.
        """
        while self.running:
            try:
                if not self.vlm_processing_queue.empty():
                    back_object = self.vlm_processing_queue.get()
                    # Process the back object with VLM
                    self.logger.info(f"Processing BackObject {back_object.id} for VLM.")
                    best_frame = back_object.best_frame
                    if best_frame is not None:
                        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_save_path  = os.path.join(self.Capture_save_folder, f"{current_time}.jpg")
                        cv2.imwrite(image_save_path, best_frame)    
                        
                    # Add your VLM processing logic here
            except Exception as e:
                self.logger.error(f"Error in VLM processing thread: {e} | {traceback.format_exc()}")