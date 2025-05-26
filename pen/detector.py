# import onnxruntime as ort
# import numpy as np
# import cv2

# class PenDetector:
#     def __init__(self, model_path=r'E:\aPenproject\Apen-project\video_processor\pen\best.onnx', conf_thres=0.25):
#         print(f"Loading ONNX model from: {model_path}")
#         self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
#         self.conf_thres = conf_thres

#         # 获取输入输出名
#         self.input_name = self.session.get_inputs()[0].name
#         self.output_name = self.session.get_outputs()[0].name

#     def detect(self, frame):
#         orig_h, orig_w = frame.shape[:2]
#         input_size = 640  # 模型要求输入尺寸
#         img = cv2.resize(frame, (input_size, input_size))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_norm = img_rgb.astype(np.float32) / 255.0
#         img_trans = np.transpose(img_norm, (2, 0, 1))
#         img_input = np.expand_dims(img_trans, axis=0)

#         outputs = self.session.run([self.output_name], {self.input_name: img_input})[0]
#         detections = outputs[0]

#         for det in detections:
#             conf = det[4]
#             if conf < self.conf_thres:
#                 continue

#             keypoints = []
#             for i in range((len(det) - 5) // 3):
#                 x_kpt = det[5 + i * 3]
#                 y_kpt = det[5 + i * 3 + 1]
#                 conf_kpt = det[5 + i * 3 + 2]
#                 keypoints.append((x_kpt, y_kpt, conf_kpt))

#             if len(keypoints) == 0 or keypoints[0][2] < 0.2:
#                 continue

#             # 将 640x640 坐标映射回原图
#             kpt_x, kpt_y = keypoints[0][0], keypoints[0][1]
#             x = int(kpt_x * orig_w / input_size)
#             y = int(kpt_y * orig_h / input_size)
#             return (x, y), 5

#         return None, None



import cv2
from utils.fps_controller import FPSCounter
import time
from ultralytics import YOLO
import numpy as np

class PenDetector:
<<<<<<< HEAD
    def __init__(self, model_path=r'D:\zhs\Apen-project\video_processor\pen\bestva.pt', conf_thres=0.25):
        print(f"Loading PyTorch model from: {model_path}")
=======
    def __init__(self, model_path = r'D:\zhs\Apen-project\video_processor\pen\best.pt', conf_thres=0.25):
        print(f"Loading model from: {model_path}")
>>>>>>> 61d800b00884df2631d4334f5c3eda76d63d52eb
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.conf_thres, verbose=False)
        if not results or not hasattr(results[0], "keypoints"):
            return None
        keypoints = results[0].keypoints
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return None
        pen_kps = keypoints.xy
        if len(pen_kps[0]) == 0:
            return None
        pen_tip = pen_kps[0][0].cpu().numpy()  # 第一个目标的第一个关键点
        x, y = int(pen_tip[0]), int(pen_tip[1])
        return (x, y)

def test_camera_with_pen_detection(camera_index=0, duration=15):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    print("开始测试摄像头并进行笔尖检测...")

    fps_counter = FPSCounter()
    detector = PenDetector()

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            break

        fps_counter.update()

        # 进行笔尖检测
        pen_pos = detector.detect(frame)
        if pen_pos is not None:
            cv2.circle(frame, pen_pos, 10, (0, 0, 255), 2)  # 画红色圆圈标记笔尖

        frame_count += 1
        elapsed_time = time.time() - start_time

        # 显示FPS
        cv2.putText(frame, f"FPS: {fps_counter.fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera with Pen Detection", frame)

        if elapsed_time > duration or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"总帧数: {frame_count}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均 FPS: {frame_count / total_time:.2f}")

if __name__ == "__main__":
    test_camera_with_pen_detection(camera_index=0, duration=15)