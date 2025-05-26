import cv2
from utils.fps_controller import FPSCounter
import time
from ultralytics import YOLO
import numpy as np
import keyboard  # pip install keyboard
from utils.ocr_processor import OCRProcessor  # 导入 OCR 处理模块

class PenDetector:
    def __init__(self, model_path=r'D:\zhs\Apen-project\video_processor\pen\bestva.pt', conf_thres=0.25):
        print(f"Loading PyTorch model from: {model_path}")
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

def test_camera_with_pen_detection_and_ocr(camera_index=0, duration=60):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("开始测试摄像头并进行按空格框选笔尖...")

    fps_counter = FPSCounter()
    detector = PenDetector()
    ocr_processor = OCRProcessor()  # 初始化 OCR 处理器

    frame_count = 0
    start_time = time.time()

    space_was_pressed = False
    press_pos = None
    release_pos = None
    region = None
    ocr_result = None  # 用于存储 OCR 结果

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取帧失败")
            break

        fps_counter.update()

        space_is_pressed = keyboard.is_pressed('space')

        # 按下空格，检测并记录起点（第一次检测有效笔尖位置）
        if space_is_pressed and not space_was_pressed:
            pos = detector.detect(frame)
            if pos is not None:
                press_pos = pos
                print(f"空格按下，起点: {press_pos}")

        # 抬起空格，检测并记录终点，计算框选区域
        if not space_is_pressed and space_was_pressed:
            pos = detector.detect(frame)
            if pos is not None:
                release_pos = pos
                print(f"空格抬起，终点: {release_pos}")

                if press_pos is not None and release_pos is not None:
                    x1, y1 = press_pos
                    x2, y2 = release_pos
                    # 计算左上角和宽高
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2 - x1), abs(y2 - y1)
                    region = (x, y, w, h)
                    print(f"选中区域: {region}")

                    # 对框选区域进行 OCR 识别
                    if w > 0 and h > 0:
                        # 裁剪框选区域
                        roi = frame[y:y+h, x:x+w]
                        # 进行 OCR 识别
                        boxes, texts, scores = ocr_processor.process_image(roi)
                        ocr_result = (boxes, texts, scores)
                        print("OCR 结果:")
                        for text, score in zip(texts, scores):
                            print(f"文字: {text}, 置信度: {score:.2f}")
                else:
                    print("起点或终点无效，无法设置区域")

        space_was_pressed = space_is_pressed

        # 画选中区域框
        if region is not None:
            x, y, w, h = region
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示 FPS
        cv2.putText(frame, f"FPS: {fps_counter.fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 如果有 OCR 结果，在图像上绘制
        if ocr_result is not None and region is not None:
            x, y, w, h = region
            boxes, texts, scores = ocr_result
            # 将 OCR 结果绘制到原始图像上（需要调整坐标）
            adjusted_boxes = []
            for box in boxes:
                adjusted_box = []
                for point in box:
                    # 将 OCR 坐标从 ROI 坐标系转换为原始图像坐标系
                    adjusted_point = (point[0] + x, point[1] + y)
                    adjusted_box.append(adjusted_point)
                adjusted_boxes.append(adjusted_box)
            # 绘制 OCR 结果
            ocr_frame = ocr_processor.draw_results(frame, adjusted_boxes, texts, scores)
            cv2.imshow("Camera with Pen Detection and OCR", ocr_frame)
        else:
            cv2.imshow("Camera with Pen Detection and OCR", frame)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > duration or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_with_pen_detection_and_ocr(camera_index=0, duration=10000)