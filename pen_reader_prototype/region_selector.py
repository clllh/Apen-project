import cv2

class RegionSelector:
    def __init__(self):
        self.start_point = None
        self.end_point = None
    
    def set_start_point(self, point):
        self.start_point = point
    
    def set_end_point(self, point):
        self.end_point = point
    
    def get_selection_rect(self):
        """获取标准化矩形区域(x,y,w,h)"""
        if self.start_point and self.end_point:
            x = min(self.start_point[0], self.end_point[0])
            y = min(self.start_point[1], self.end_point[1])
            w = abs(self.start_point[0] - self.end_point[0])
            h = abs(self.start_point[1] - self.end_point[1])
            return (x, y, w, h)
        return None
    
    def draw_selection(self, frame):
        """在帧上绘制选择区域"""
        rect = self.get_selection_rect()
        if rect:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)