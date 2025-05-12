import cv2
from typing import Optional, Tuple
import numpy as np

class RegionSelector:
    def __init__(self):
        self.start = None
        self.end = None
        self.is_selecting = False
    
    def begin_selection(self, point: tuple) -> None:
        self.start = point
        self.is_selecting = True
    
    def update_selection(self, point: tuple) -> None:
        if self.is_selecting:
            self.end = point
    
    def end_selection(self) -> None:
        self.is_selecting = False
    
    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        if self.start and self.end:
            x = min(self.start[0], self.end[0])
            y = min(self.start[1], self.end[1])
            w = abs(self.start[0] - self.end[0])
            h = abs(self.start[1] - self.end[1])
            return (x, y, w, h)
        return None
    
    def draw(self, frame: np.ndarray) -> None:
        if self.is_selecting and self.start and self.end:
            cv2.rectangle(frame, self.start, self.end, (0, 255, 0), 2)