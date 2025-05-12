"""
视频处理核心包
包含摄像头控制、帧缓冲和实用工具
"""

# 版本信息
__version__ = "1.0.0"

# 暴露主要接口
from .camera.frame_capturer import VideoCapturer
from .camera.frame_buffer import FrameBuffer
from .utils.fps_controller import FPSCounter, FPSLimiter

# 包级别初始化代码
def init_hardware():
    """检查OpenCV可用性"""
    import cv2
    print(f"OpenCV version: {cv2.__version__}")

__all__ = ['VideoCapturer', 'FrameBuffer', 'FPSCounter', 'FPSLimiter']