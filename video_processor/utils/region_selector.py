import cv2
from typing import Optional, Tuple
import numpy as np

class RegionSelector:
    def __init__(self):
        self.start = None
        self.end = None
        self.is_selecting = False
        self._region = None  # 存储最终框选区域

    def begin_selection(self, point: tuple) -> None:
        self.start = point
        self.is_selecting = True

    def update_selection(self, point: tuple) -> None:
        if self.is_selecting:
            self.end = point

    def end_selection(self) -> None:
        if self.start and self.end:
            x = min(self.start[0], self.end[0])
            y = min(self.start[1], self.end[1])
            w = abs(self.start[0] - self.end[0])
            h = abs(self.start[1] - self.end[1])
            self._region = (x, y, w, h)
        self.is_selecting = False

    def set_region_by_two_points(self, p1: tuple, p2: tuple) -> None:
        """使用两个点直接设置区域框选（适用于按下/抬起场景）"""
        x = min(p1[0], p2[0])
        y = min(p1[1], p2[1])
        w = abs(p2[0] - p1[0])
        h = abs(p2[1] - p1[1])
        self._region = (x, y, w, h)
        self.start = p1
        self.end = p2
        self.is_selecting = False

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        return self._region

    def draw(self, frame: np.ndarray) -> None:
        # 绘制当前正在选择的区域，或已完成的区域
        if self.is_selecting and self.start and self.end:
            cv2.rectangle(frame, self.start, self.end, (0, 255, 0), 2)
        elif self._region:
            x, y, w, h = self._region
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
