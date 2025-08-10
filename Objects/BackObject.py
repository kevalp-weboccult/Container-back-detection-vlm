"""
Back Object Module.

This module provides a BackObject class for managing and tracking object instances across frames.
It maintains object state, bounding box history, confidence tracking, and best frame capture.

Classes:
    BackObject: Main class for object instance management and tracking.
"""

from datetime import datetime, timezone
import os 
import time
import traceback
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np
from Managers.ConfigManager import ConfigManager
from Modules.CustomLogger import CustomLogger
class BackObject:
    """
    Object tracking and state management class.
    
    This class maintains state information for a single detected object across multiple frames.
    It tracks bounding box history, maintains the best confidence detection, and stores the best
    frame capture for quality tracking.
    
    Attributes:
        name (str): Unique identifier name for this object instance.
        id (Optional[str]): Unique object ID for tracking across frames.
        config_manager (ConfigManager): Configuration manager instance.
        logger (logging.Logger): Logger instance for this object.
        bboxes (List[List[float]]): History of bounding boxes for this object.
        is_roi_crossed (bool): Flag indicating if object has crossed ROI.
        best_conf (Optional[float]): Highest confidence score achieved.
        best_bbox (Optional[List[float]]): Bounding box with highest confidence.
        best_frame (Optional[np.ndarray]): Frame with highest confidence detection.
        best_frame_size (Optional[Tuple[int, int]]): Dimensions of the best frame.
    """
    
    def __init__(self, name: str = "back_object", id: Optional[str] = None) -> None:
        """
        Initialize the BackObject with configuration settings.
        
        Args:
            name (str): Name of the back object instance. Defaults to "back_object".
            id (Optional[str]): Unique object ID for tracking. Defaults to None.
            
        Raises:
            Exception: If initialization fails due to configuration or logging issues.
        """
        try:
            self.name: str = name
            self.id: Optional[str] = id
            self.config_manager: ConfigManager = ConfigManager.get_instance()
            
            custom_logger: CustomLogger = CustomLogger(self.name)
            self.logger: logging.Logger = custom_logger.get_logger(
                log_file=f"logs/{name}.log",
                log_level=self.config_manager.get("LOG_LEVEL"),
                log_to_console=self.config_manager.get("LOG_TO_CONSOLE"),
            )
            self.logger.info(f"BackObject {self.name} initialized with settings: {self.config_manager.defaults}")
            
            self.bboxes: List[List[float]] = []
            self.is_roi_crossed: bool = False
            self.best_conf: Optional[float] = None
            self.best_bbox: Optional[List[float]] = None
            self.best_frame: Optional[np.ndarray] = None
            self.best_frame_size: Optional[Tuple[int, int]] = None
            self.missing_count:int = 0
            self.MIN_IMAGE_SIZE: int = self.config_manager.get("MIN_IMAGE_SIZE", 100)
            
        except Exception as e:
            print(f"Error initializing BackObject: {e} | {traceback.format_exc()}")

    def update(self, bbox: List[float], conf: float, frame: np.ndarray) -> None:
        """
        Update object state with new detection information.
        
        Updates the object's bounding box history, and if the new detection has higher
        confidence than the current best, updates the best detection information.
        
        Args:
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2].
            conf (float): Confidence score for this detection (0.0 to 1.0).
            frame (np.ndarray): Frame array containing this detection.
            
        Raises:
            Exception: If an error occurs during state update.
        """
        try:
            if len(self.bboxes) > 20:
                self.bboxes.pop(0)
            self.bboxes.append(bbox)
            
            frame_size: Tuple[int, int] = (frame.shape[0], frame.shape[1])
            
            is_valid_size = (frame_size[0] > self.MIN_IMAGE_SIZE and 
                           frame_size[1] > self.MIN_IMAGE_SIZE and 
                           frame_size[0] < frame_size[1] * 1.5)
            
            should_update = (self.best_frame is None or 
                           (conf > self.best_conf and 
                            frame_size[0] > self.best_frame_size[0] and 
                            frame_size[1] > self.best_frame_size[1]))
            
            if is_valid_size and should_update:
                self.best_frame = frame
                self.best_frame_size = frame_size
                self.best_conf = conf
                self.best_bbox = bbox
                    
                    
            
        except Exception as e:
            self.logger.error(f"Error updating BackObject: {e} | {traceback.format_exc()}")