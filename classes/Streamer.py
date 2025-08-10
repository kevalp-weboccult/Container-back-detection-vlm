"""
Video Streamer Module.

This module provides a Streamer class for handling video stream capture and processing.
It manages video input from cameras or video files, provides frame metadata, and handles
stream restarts and error recovery. The Streamer integrates with queues for efficient
frame processing in multi-threaded applications.

Classes:
    Streamer: Main class for video stream capture and management.

Example:
    >>> streamer = Streamer(name="camera1", camera_url=0)
    >>> streamer.start()
    >>> frame_data = streamer.get_metadata()
    >>> streamer.stop()
"""

import os
import time
import traceback
from typing import Union, Dict, Any, Optional
import cv2
from ffmpegcv import VideoCaptureStream
from Managers.ConfigManager import ConfigManager
from Modules.CustomLogger import CustomLogger
from queue import Queue
import logging

class Streamer:
    """
    Video stream capture and management class.
    
    This class handles video stream capture from cameras or video files, provides
    frame metadata, manages stream restarts, and integrates with processing queues.
    It includes error recovery mechanisms and configurable frame rate control.
    
    Attributes:
        name (str): Name identifier for the streamer instance.
        settings_manager (ConfigManager): Configuration manager instance.
        logger (logging.Logger): Logger instance for this streamer.
        stream (cv2.VideoCapture): OpenCV video capture object.
        height (int): Frame height in pixels.
        width (int): Frame width in pixels.
        fps (int): Frames per second from the video source.
        ret_false_count (int): Counter for failed frame reads.
        frame_count (int): Total number of frames processed.
        writer_queue (Queue[Dict[str, Any]]): Queue for frame data output.
        max_ret_false_count (int): Maximum allowed consecutive failed reads.
        is_live (bool): Whether the stream is live or recorded.
        running (bool): Flag indicating if the streamer is active.
        CUSTOM_FPS (bool): Whether to use custom frame rate control.
        FPS (int): Custom frame rate setting.
    """
    
    def __init__(self, name: str = "streamer", camera_url: Union[str, int] = 0) -> None:
        """
        Initialize the Streamer with video source and configuration.
        
        Args:
            name (str): Name identifier for the streamer instance. Defaults to "streamer".
            camera_url (Union[str, int]): Camera index (int) or video file path (str).
                                        Defaults to 0 (first camera).
                                        
        Raises:
            Exception: If initialization fails due to configuration or camera access issues.
        """
        try:
            self.name: str = name
            self.settings_manager: ConfigManager = ConfigManager.get_instance()
            
            custom_logger: CustomLogger = CustomLogger(self.name)
            self.logger: logging.Logger = custom_logger.get_logger(
                log_file=f"logs/{name}.log",
                log_level=self.settings_manager.get("LOG_LEVEL"),
                log_to_console=self.settings_manager.get("LOG_TO_CONSOLE"),
            )
            self.USE_FFMPEG: bool = self.settings_manager.get("USE_FFMPEG", True)
            self.logger.info(f"Streamer {self.name} initialized with settings: {self.settings_manager.defaults}")
            
            self.stream = VideoCaptureStream(camera_url) if self.USE_FFMPEG else cv2.VideoCapture(camera_url)
            # self.height: int = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # self.width: int = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            # self.fps: int = int(self.stream.get(cv2.CAP_PROP_FPS))
            
            # Stream state tracking
            self.ret_false_count: int = 0
            self.frame_count: int = 0
            
            # Queue and configuration
            self.writer_queue: Queue[Dict[str, Any]] = Queue(maxsize=1)
            self.max_ret_false_count: int = self.settings_manager.get("MAX_RET_FALSE_COUNT", 10)
            self.is_live: bool = self.settings_manager.get("IS_LIVE", True)
            self.running: bool = True
            
            # Frame rate control
            self.CUSTOM_FPS: bool = self.settings_manager.get("CUSTOM_FPS", False)
            self.FPS: int = self.settings_manager.get("FPS", 5)

        except Exception as e:
            print(f"Error initializing Streamer: {e} | {traceback.format_exc()}")
    

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Capture a frame from the video stream and return it with metadata.
        
        Reads a frame from the video stream and packages it with frame count
        and other metadata. Handles stream errors and restart logic.
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing frame data and metadata,
                                    or None if frame capture fails.
                                    Dictionary keys:
                                    - 'frame': np.ndarray - The captured frame
                                    - 'frame_count': int - Sequential frame number
                                    
        Raises:
            Exception: If an error occurs during frame capture or processing.
        """
        try:
            frame_data: Dict[str, Any] = {}
            ret: bool
            frame: Any
            ret, frame = self.stream.read()
            
            if not ret:
                self.logger.error("Failed to read frame from stream")
                self.ret_false_count += 1
                if self.ret_false_count >= self.max_ret_false_count:
                    self.logger.warning(f"Max ret false count reached: {self.ret_false_count}. Restarting stream.")
                    self.restart_stream()
                return None
                
            self.frame_count += 1
            frame_data['frame'] = frame
            
            if self.frame_count == 1:
                cv2.imwrite(f"first_frame.jpg", frame)
                
            frame_data['frame_count'] = self.frame_count
            return frame_data
            
        except Exception as e:
            self.logger.error(f"Error getting frame: {e} | {traceback.format_exc()}")
            return None
    
    def _read_frame_with_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Internal method to read frame with metadata.
        
        Wrapper around get_metadata() for internal use with additional
        error handling and logging.
        
        Returns:
            Optional[Dict[str, Any]]: Frame data dictionary or None if failed.
        """
        try:
            data: Optional[Dict[str, Any]] = self.get_metadata()
            if data is None:
                self.logger.debug("No data received from get_metadata, returning None")
                return None
            return data
        except Exception as e:
            self.logger.error(f"Error reading frame with metadata: {e} | {traceback.format_exc()}")
            return None
    
    
    
    def restart_stream(self) -> None:
        """
        Restart the video stream after connection failure.
        
        Releases the current stream, waits for stabilization, and reopens
        the video capture with the configured camera URL. Resets counters
        and stream properties.
        
        Raises:
            Exception: If stream restart fails.
        """
        try:
            self.stream.release()
            time.sleep(3)
            camera_url: Union[str, int] = self.settings_manager.get("CAMERA_URL", 0)
            self.stream = cv2.VideoCapture(camera_url)
            # self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            # self.fps = int(self.stream.get(cv2.CAP_PROP_FPS))
            self.ret_false_count = 0
            self.frame_count = 0
            self.logger.info("Stream restarted successfully")
        except Exception as e:
            self.logger.error(f"Error restarting stream: {e} | {traceback.format_exc()}")

    def start(self) -> None:
        """
        Start the main streaming loop.
        
        Continuously captures frames from the video stream and places them
        in the writer queue. Handles custom frame rate control, queue management,
        and error recovery. Runs until the running flag is set to False.
        
        The loop includes:
        - Frame capture with metadata
        - Custom FPS filtering (if enabled)
        - Queue management for downstream processing
        - Error handling and recovery
        """
        while self.running:
            try:
                frame_data: Optional[Dict[str, Any]] = self._read_frame_with_metadata()
                if frame_data is None:
                    self.logger.debug(f"No frame data received from stream. Ret False Count: {self.ret_false_count}")
                    time.sleep(0.1)
                    continue
                
                self.logger.debug(f"Frame Data: {frame_data['frame_count']}")

                if not self.is_live:
                    while self.writer_queue.full():
                        time.sleep(0.1)
                
                if not self.writer_queue.full():
                    if self.CUSTOM_FPS:
                        fps_condition: bool = self.frame_count % self.FPS == 0
                        print(f"{fps_condition} | Frame Count: {self.frame_count} | FPS: {self.FPS}")
                        if not fps_condition:
                            continue
                    self.writer_queue.put(frame_data)
                    self.logger.debug(f"Frame {frame_data['frame_count']} sent to writer queue")
  
            except Exception as e:
                self.logger.error(f"Error in stream loop: {e} | {traceback.format_exc()}")
                time.sleep(1)

    def __info__(self) -> None:
        """
        Log comprehensive information about the streamer state.
        
        Logs all important streamer properties including configuration,
        stream properties, and current state for debugging purposes.
        """
        self.logger.info("*" * 20)
        self.logger.info(f"Streamer Name: {self.name}")
        self.logger.info(f"Camera URL: {self.settings_manager.get('CAMERA_URL', 0)}")
        # self.logger.info(f"Height: {self.height}")
        # self.logger.info(f"Width: {self.width}")
        # self.logger.info(f"FPS: {self.fps}")
        self.logger.info(f"Frame Count: {self.frame_count}")
        self.logger.info(f"Is Live: {self.is_live}")
        self.logger.info(f"Max Ret False Count: {self.max_ret_false_count}")
        self.logger.info(f"Running: {self.running}")
        self.logger.info("*" * 20)
    
    def get_queue_size(self) -> None:
        """
        Log the current size of the writer queue.
        
        Useful for monitoring queue utilization and potential bottlenecks
        in the processing pipeline.
        """
        self.logger.info(f"Streamer Writer Queue Size: {self.writer_queue.qsize()}")

    def release(self) -> None:
        """
        Release the video stream resources.
        
        Safely closes the video capture object if it's currently opened.
        Should be called before destroying the streamer instance.
        
        Raises:
            Exception: If an error occurs during stream release.
        """
        try:
            if self.stream.isOpened():
                self.stream.release()
                self.logger.info("Stream released successfully")
            else:
                self.logger.warning("Stream was not opened, nothing to release")
        except Exception as e:
            self.logger.error(f"Error releasing stream: {e} | {traceback.format_exc()}")
    
    def stop(self) -> None:
        """
        Stop the streamer and release all resources.
        
        Sets the running flag to False to stop the main loop, releases
        the video stream, and logs the shutdown process.
        
        Raises:
            Exception: If an error occurs during shutdown.
        """
        try:
            self.running = False
            self.release()
            time.sleep(0.2)
            self.logger.info("Streamer stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping streamer: {e} | {traceback.format_exc()}")
            
    def __str__(self) -> str:
        """
        Return a string representation of the Streamer instance.
        
        Returns:
            str: Formatted string containing key streamer properties and state.
        """
        camera_url: Union[str, int] = self.settings_manager.get('CAMERA_URL', 0)
        return (f"Streamer(name={self.name}, camera_url={camera_url}, "
                f"frame_count={self.frame_count}, is_live={self.is_live}, "
                f"max_ret_false_count={self.max_ret_false_count}, running={self.running})")