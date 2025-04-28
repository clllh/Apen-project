import numpy as np
from collections import deque

class PenTracker:
    def __init__(self, buffer_size=5):
        self.positions = deque(maxlen=buffer_size)
        self.speeds = deque(maxlen=buffer_size)
    
    def update(self, position: tuple) -> None:
        if position:
            if self.positions:
                prev = self.positions[-1]
                dx = position[0] - prev[0]
                dy = position[1] - prev[1]
                self.speeds.append(np.sqrt(dx**2 + dy**2))
            self.positions.append(position)
    
    def get_speed(self) -> float:
        return np.mean(self.speeds) if self.speeds else 0.0
    
    def is_moving(self, threshold=5.0) -> bool:
        return self.get_speed() > threshold