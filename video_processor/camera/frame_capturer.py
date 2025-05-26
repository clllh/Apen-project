#fram_capture.py
import cv2
from .frame_buffer import FrameBuffer
from typing import Optional, Tuple
import numpy as np
import threading
import time

class VideoCapturer:
    def __init__(self, src=0, target_fps=30):
        """
        参数:
            src: 摄像头ID或视频路径
            target_fps: 目标帧率 (15-30)
        """
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.frame_buffer = FrameBuffer()
        self.target_fps = target_fps
        self._setup_camera()

    def _setup_camera(self):
        """配置摄像头参数"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
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

class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480, target_fps=30):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)

        self.frame_buffer = FrameBuffer(buffer_size=2)

        self.ret = False
        self.stopped = True
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.stopped:
            self.stopped = False
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame_buffer.add_frame(frame)
            # 这里稍作延时，防止占用过高CPU
            time.sleep(0.001)

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            if not self.ret:
                return False, None, None
            prev_frame, curr_frame = self.frame_buffer.get_frames()
            return True, prev_frame, curr_frame

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
        self.cap.release()
        
    def release(self):
        self.stopped = True
        time.sleep(0.1)  # 给线程时间退出
        self.cap.release()