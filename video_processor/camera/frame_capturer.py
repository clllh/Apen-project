import cv2
from .frame_buffer import FrameBuffer
from typing import Optional, Tuple
import numpy as np

class VideoCapturer:
    def __init__(self, src=0, target_fps=30):
        """
        参数:
            src: 摄像头ID或视频路径
            target_fps: 目标帧率 (15-30)
        """
        self.cap = cv2.VideoCapture(src)
        self.frame_buffer = FrameBuffer()
        self.target_fps = target_fps
        self._setup_camera()

    def _setup_camera(self):
        """配置摄像头参数"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # 注意：实际帧率取决于硬件支持
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)  

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取帧对
        返回: (status, prev_frame, curr_frame)
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None
        
        self.frame_buffer.add_frame(frame)
        prev, curr = self.frame_buffer.get_frames()
        return True, prev, curr

    def release(self):
        self.cap.release()