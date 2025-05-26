#frame_buffer.py
import numpy as np
from collections import deque

class FrameBuffer:
    def __init__(self, buffer_size=2):
        """双帧环形缓冲区"""
        self.buffer = deque(maxlen=buffer_size)
    
    def add_frame(self, frame: np.ndarray):
        """添加帧到缓冲区"""
        self.buffer.append(frame.copy())
    
    def get_frames(self) -> tuple:
        """
        获取帧对 (previous, current)
        返回: (None, current) 当只有一帧时
        """
        if len(self.buffer) == 2:
            return self.buffer[0], self.buffer[1]
        elif len(self.buffer) == 1:
            return None, self.buffer[0]
        return None, None

    def clear(self):
        self.buffer.clear()