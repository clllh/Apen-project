from ultralytics import YOLO
import cv2
import numpy as np

class PenDetector:
    def __init__(self, model_path='best.pt', conf_thres=0.25):
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        # 运行 YOLO pose 模型推理
        results = self.model.predict(source=frame, conf=self.conf_thres, verbose=False)

        if not results or not results[0].keypoints:
            return None, None

        keypoints = results[0].keypoints.xy  # Shape: (N, K, 2)

        if keypoints is None or len(keypoints) == 0:
            return None, None

        # 只考虑第一只笔（第一个检测框）
        pen_kps = keypoints[0]  # K x 2
        if len(pen_kps) == 0:
            return None, None

        # 关键点 0 是笔尖（你的训练数据中设置的）
        pen_tip = pen_kps[0]  # (x, y)

        x, y = int(pen_tip[0]), int(pen_tip[1])
        return (x, y), 5  # 第二个返回值为显示用的笔尖“半径”

# import cv2
# import numpy as np

# # class PenDetector:
# #     def __init__(self, lower_red=(0, 120, 70), upper_red=(10, 255, 255)):
# #         """初始化红色笔尖检测器"""
# #         self.lower_red = np.array(lower_red)
# #         self.upper_red = np.array(upper_red)
# #         self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
# #     def detect(self, frame):
# #         """检测红色笔尖位置"""
# #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
# #         # 红色检测（处理HSV色环的0°和180°附近）
# #         mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
# #         mask2 = cv2.inRange(hsv, 
# #                           np.array([170, 120, 70]), 
# #                           np.array([180, 255, 255]))
# #         mask = cv2.bitwise_or(mask1, mask2)
        
# #         # 形态学处理
# #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
# #         # 寻找最大轮廓
# #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #         if contours:
# #             max_contour = max(contours, key=cv2.contourArea)
# #             ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
# #             return (int(x), int(y)), radius
# #         return None, 0

# #     def draw_marker(self, frame, position, radius):
# #         """在帧上绘制笔尖标记"""
# #         if position:
# #             cv2.circle(frame, position, int(radius), (0, 255, 0), 2)
# #             cv2.circle(frame, position, 5, (0, 0, 255), -1)

