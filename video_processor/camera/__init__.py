"""
摄像头控制模块
包含视频捕获和帧缓冲功能
"""

# 子模块导出
from .frame_capturer import VideoCapturer
from .frame_buffer import FrameBuffer

__all__ = ['VideoCapturer', 'FrameBuffer']