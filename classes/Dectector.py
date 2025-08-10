"""
This module defines the Detector class, which uses a YOLO model to perform object detection on video frames.
It initializes the YOLO model, processes frames from a reader queue, and draws detections on the frames.
It also manages logging and configuration settings.


"""
import os
from pathlib import Path
import time
import traceback
from typing import Union, List, Dict, Optional, Any
import numpy as np
import cv2
from Managers.ConfigManager import ConfigManager
from Modules.CustomLogger import CustomLogger
from ultralytics import YOLO
from queue import Queue
import logging


class Detector:
    def __init__(self, name: str = "detector", yolo_model: Optional[str] = None) -> None:
        """
        Initialize the Detector with a YOLO model and configuration settings.
        If no model is provided, it defaults to "yolov8n.pt".
        
        Args:
            name (str): Name of the detector instance.
            yolo_model (Optional[str]): Path to the YOLO model file. Defaults to "yolov8n.pt".
        """
        try:
            self.name: str = name
            self.yolo_model: str = yolo_model if yolo_model is not None else "yolov8n.pt"
            
            self.config_manager: ConfigManager = ConfigManager.get_instance()
            custom_logger: CustomLogger = CustomLogger(self.name)
            self.logger: logging.Logger = custom_logger.get_logger(
                log_file=f"logs/{name}.log",
                log_level=self.config_manager.get("LOG_LEVEL"),
                log_to_console=self.config_manager.get("LOG_TO_CONSOLE"),
            )
            self.logger.info(f"Detector {self.name} initialized with settings: {self.config_manager.defaults}")
            
            self.model: YOLO = YOLO(self.yolo_model)
            self.running: bool = True
            self.writer_queue: Queue[Dict[str, Any]] = Queue(maxsize=1)
            self.reader_queue: Optional[Queue[Dict[str, Any]]] = None
            
            # Configuration parameters with type hints
            self.MODEL_IMGSZ: List[int] = self.config_manager.get("MODEL_IMGSZ", [640, 640])
            self.is_live: bool = self.config_manager.get("IS_LIVE", True)
            self.confidence: float = self.config_manager.get("CONFIDENCE", 0.5)
            self.iou: float = self.config_manager.get("IOU", 0.45)
            
            self.inference_frame: Optional[np.ndarray] = None
        
        except Exception as e:
            print(f"Error initializing Detector: {e} | {traceback.format_exc()}")
    

    def draw_detections_on_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detections on frame.
        
        Args:
            frame (np.ndarray): The input frame to draw on.
            detections (List[Dict[str, Any]]): List of detection dictionaries.
            
        Returns:
            np.ndarray: Frame with detections drawn on it.
        """
        for detection in detections:
            bbox: List[int] = detection["bbox"]
            class_id: int = detection["class_id"]
            conf: float = detection["confidence"]
            text: str = f"{class_id} | {conf:.2f}"
            cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            
        return frame
    
    def get_detection_data(self, frame: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """
        Get detection data from the frame using YOLO model.
        
        Args:
            frame (np.ndarray): Input frame for detection.
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of detections with bbox, confidence, and class_id.
                                           Returns empty list if no detections found.
                                           Returns None if an error occurs.
        """
        try:
            detections: List[Dict[str, Any]] = []
            results = self.model.predict(
                frame, 
                iou=self.iou, 
                conf=self.confidence, 
                verbose=True, 
                imgsz=self.MODEL_IMGSZ,
                classes=0
            )[0]
            self.inference_frame = frame.copy()
            bbox_datas = results.boxes.data
            self.logger.info(f"Detected {len(bbox_datas)} objects in the frame.")
            
            for bbox_data in bbox_datas:
                bbox_data_np: np.ndarray = bbox_data.cpu().numpy()
                if bbox_data_np[4] < self.confidence:
                    continue
                x1, y1, x2, y2, conf, cls = bbox_data_np
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                detection: Dict[str, Any] = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": int(cls),
                }
                detections.append(detection)
            
            return detections

        except Exception as e:
            self.logger.error(f"Error getting detection data: {e} | {traceback.format_exc()}")
            return None

    def start(self, reader_queue: Queue[Dict[str, Any]]) -> None:
        """
        Start the detector processing loop.
        
        Args:
            reader_queue (Queue[Dict[str, Any]]): Queue containing frame data to process.
        """
        try:
            self.reader_queue = reader_queue
            self.logger.info(f"Detector {self.name} started with reader queue: {self.reader_queue}")
            
            while self.running:
                if self.reader_queue.empty():
                    time.sleep(0.1)
                    continue
                    
                frame_data: Optional[Dict[str, Any]] = self.reader_queue.get()
                if frame_data is None:
                    time.sleep(0.1)
                    continue
                    
                frame: Optional[np.ndarray] = frame_data.get('frame')
                if frame is None:
                    self.logger.warning("Received None frame from reader queue, skipping...")
                    continue
                    
                detections: Optional[List[Dict[str, Any]]] = self.get_detection_data(frame)
                frame_data['detections'] = detections
                
                if self.inference_frame is not None:
                    frame_count: int = frame_data.get('frame_count', 0)
                    cv2.putText(
                        self.inference_frame, 
                        f"{frame_count}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2
                    )
                    if detections is not None:
                        self.inference_frame = self.draw_detections_on_frame(self.inference_frame, detections)
                    frame_data['inference_frame'] = self.inference_frame
                
                if not self.is_live:
                    while self.writer_queue.full():
                        time.sleep(0.1)
                        
                if not self.writer_queue.full():
                    self.writer_queue.put(frame_data)
                    
        except Exception as e:
            self.logger.error(f"Error in Detector start method: {e} | {traceback.format_exc()}")
    

    def stop(self) -> None:
        """
        Stop the detector, releasing resources.
        """
        try:
            self.running = False
            self.logger.info(f"Detector {self.name} stopped.")
            # if self.model:
            #     self.model.close()
        except Exception as e:
            self.logger.error(f"Error stopping Detector: {e} | {traceback.format_exc()}")
