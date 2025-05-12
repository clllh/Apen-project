from ultralytics import YOLO
import cv2
import numpy as np

class PenDetector:
    def __init__(self, model_path = r'D:\zhs\Apen-project\video_processor\pen\best.pt', conf_thres=0.25):
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