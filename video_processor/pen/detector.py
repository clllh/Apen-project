import cv2
import numpy as np
from typing import Optional, Tuple

class PenDetector:
    def __init__(self, hsv_lower=(0, 120, 70), hsv_upper=(10, 255, 255)):
        self.lower = np.array(hsv_lower)
        self.upper = np.array(hsv_upper)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        """返回笔尖位置和半径"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
            return ((int(x), int(y)), float(radius))
        return None