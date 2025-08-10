import os 
import time
import traceback
from typing import Dict, Any, Optional, Union, List,TYPE_CHECKING
import json
from Modules.CustomLogger import CustomLogger
from Managers.ConfigManager import ConfigManager
from classes.Streamer import Streamer
from classes.Dectector import Detector
if TYPE_CHECKING:
    from Managers.CameraManager import CameraManager
from Managers.BackManager import BackManager
from byte_tracker_pytorch.byte_tracker_model import BYTETracker
from threading import Thread
from queue import Queue
class CameraObject:
    def __init__(self, name: str = "camera_object",url=None,camera_manger=None) -> None:
        """
        
        """
        try:
            self.name: str = name
            self.url = url
            self.back_manager: Optional[BackManager] = BackManager()
            self.camera_manager: CameraManager = camera_manger
            self.config_manager: ConfigManager = ConfigManager.get_instance()
            custom_logger: CustomLogger = CustomLogger(self.name)
            self.logger = custom_logger.get_logger(
                log_file=f"logs/{name}.log",
                log_level=self.config_manager.get("LOG_LEVEL"),
                log_to_console=self.config_manager.get("LOG_TO_CONSOLE"),
            )
            self.logger.info(f"CameraObject {self.name} initialized with settings: {self.config_manager.defaults}")
            try:
                self.url:int = int(self.url)
            except ValueError:
                self.logger.error(f"Invalid URL format for camera {self.name}: {self.url}")
                self.url:str = url
            self.streamer:Streamer = Streamer(camera_url=self.url)
            self.yolo_model_path = self.config_manager.get("YOLO_MODEL_PATH", "yolov8n.pt")
            self.detector:Detector = Detector(name=f"detector", yolo_model=self.yolo_model_path)
            self.running: bool = True
            self.tracker : BYTETracker = BYTETracker(
                fps=50,
                first_track_thresh=0.1,
                second_track_thresh=0.2,
                match_thresh=0.99,
                track_buffer=50, 
                resize_width_height=640 , 
                mot20=True
            )
            self.camera_id: Optional[str] = None
            self.streamer_thread: Optional[Thread] = None   
            self.detector_thread: Optional[Thread] = None
            self.tracker_thread: Optional[Thread] = None
            self.writer_queue: Queue[Dict[str, Any]] = Queue(maxsize=1)

            
            
            
        except Exception as e:
            print(f"Error initializing CameraObject: {e} | {traceback.format_exc()}")
            print(f"Error initializing CameraObject: {e} | {traceback.format_exc()}")
    
    def start_tracking(self):
        """
        Start the tracking process for the camera object.
        """
        while self.running:
            try:
                time.sleep(0.01)
                if self.detector.writer_queue.empty():
                    time.sleep(0.05)
                    continue
                frame_data = self.detector.writer_queue.get()
                detection_data = frame_data["detections"]
                frame = frame_data["frame"]
                frame_count = frame_data["frame_count"]
                box_list = []
                id_list = []
                conf_list = []
                class_list = []
                if len(detection_data) < 1:
                    frame_data['detections'] = []
                    frame_data['ids'] = []
                    frame_data['conf'] = []
                else:
                    detections_to_track = []
                
                for det in detection_data:
                    bbox = det["bbox"]
                    conf = det['confidence']
                    bbox = [int(x) for x in bbox]
                    detections_to_track.append(
                            [
                                det["bbox"][0],
                                det["bbox"][1],
                                det["bbox"][2],
                                det["bbox"][3],
                                det["confidence"],
                                det["class_id"]
                            ]
                        )
                
                    track_list  = self.tracker.update(detections_to_track)       
                    self.logger.info(f"{len(track_list)=}")

                    
                    id_list = [t.track_id for t in track_list]  # Get id list
                    box_list = [t.tlbr for t in track_list]     # Get box list
                    conf_list = [t.score for t in track_list]   # Get conf scores
                    class_list = [t.class_name for t in track_list] # Get class list

                frame_data['detections'] = box_list
                frame_data['ids'] = id_list
                frame_data['conf'] = conf_list
                frame_data['class'] = class_list
                frame_data['frame_count'] = frame_count
                frame_data['frame'] = frame

                if not self.writer_queue.full():
                    self.writer_queue.put(frame_data)
                    self.logger.info(f"Frame data added to writer queue for {self.name}.")
                else:
                    self.logger.warning(f"Writer queue is full for {self.name}, skipping frame data.")
            




            except Exception as e:
                self.logger.error(f"Error in start_tracking for {self.name}: {e} | {traceback.format_exc()}")


    def start(self):
        try:
            self.streamer_thread = Thread(target=self.streamer.start, name=f"{self.name}_streamer_thread", daemon=True)
            self.streamer_thread.start()
            self.logger.info(f"Streamer thread started for {self.name}.")     
            self.detector_thread = Thread(target=self.detector.start, name=f"{self.name}_detector_thread", daemon=True,args=(self.streamer.writer_queue,))
            self.detector_thread.start()
            self.logger.info(f"Detector thread started for {self.name}.")
            self.tracker_thread = Thread(target=self.start_tracking, name=f"{self.name}_tracker_thread", daemon=True)
            self.tracker_thread.start()
            self.logger.info(f"Tracker thread started for {self.name}.")
            self.logger.info(f"CameraObject {self.name} started successfully.")
            self.back_manager_thread = Thread(target=self.back_manager.start, name=f"{self.name}_back_manager_thread", daemon=True,args=(self.writer_queue,))
            self.back_manager_thread.start()
            # self.vlm_processing_thread = Thread(target=self.back_manager.vlm_processing, name=f"{self.name}_vlm_processing_thread", daemon=True)
            # self.vlm_processing_thread.start()
                  
        except Exception as e:
            self.logger.error(f"Error starting CameraObject {self.name}: {e} | {traceback.format_exc()}")
            print(f"Error starting CameraObject {self.name}: {e} | {traceback.format_exc()}")
                

