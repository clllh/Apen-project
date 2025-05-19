from camera.frame_capturer import VideoCapturer
from utils.fps_controller import FPSCounter, FPSLimiter
from pen.detector import PenDetector
from utils.region_selector import RegionSelector
from utils.ocr_processor import OCRProcessor
import cv2
import keyboard
import numpy as np

def preprocess_image(image):
    # 图像预处理：灰度化、二值化、降噪等
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用高斯滤波去除噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 使用自适应阈值分割
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def main():
    # 初始化模块
    capturer = VideoCapturer(src=0, target_fps=25)
    fps_counter = FPSCounter()
    fps_limiter = FPSLimiter(target_fps=25)
    pen_detector = PenDetector()
    region_selector = RegionSelector()
    ocr_processor = OCRProcessor(lang='ch_sim')  # 初始化 EasyOCR 处理器
    ocr_active = False
    last_ocr_text = ""

    try:
        while True:
            # 控制帧率
            fps_limiter.wait()

            # 捕获双帧
            ret, prev_frame, curr_frame = capturer.read()
            if not ret:
                break

            # 更新FPS计数
            fps_counter.update()

            # 笔尖检测
            pen_result = pen_detector.detect(curr_frame)
            pen_position, pen_radius = pen_result if pen_result else (None, None)

            # 区域选择控制
            if keyboard.is_pressed('space'):
                if pen_position and not region_selector.is_selecting:
                    region_selector.begin_selection(pen_position)
            else:
                if region_selector.is_selecting and pen_position:
                    region_selector.end_selection()

            # 更新选择区域
            if region_selector.is_selecting and pen_position:
                region_selector.update_selection(pen_position)

            # 显示处理
            display_frame = curr_frame.copy()

            # 显示笔尖位置
            if pen_position:
                cv2.circle(display_frame, pen_position, int(pen_radius), (0, 0, 255), 2)
                cv2.putText(display_frame, f"Pen: {pen_position}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示选择区域
            region_selector.draw(display_frame)

            # 显示FPS
            cv2.putText(display_frame, f"FPS: {fps_counter.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Pen Reader System', display_frame)

            # 显示选定区域
            if not region_selector.is_selecting and region_selector.get_region():
                x, y, w, h = region_selector.get_region()
                selected_roi = curr_frame[y:y+h, x:x+w]
                cv2.imshow("Selected Region", selected_roi)

                # 图像预处理
                processed_roi = preprocess_image(selected_roi)

                # OCR触发逻辑
                if keyboard.is_pressed('o') and not ocr_active:
                    ocr_active = True
                    print("\n" + "="*40)
                    print("开始OCR识别...")

                    # 执行OCR
                    last_ocr_text = ocr_processor.process(processed_roi)

                    print("识别结果:")
                    print(last_ocr_text)
                    print("="*40 + "\n")

                    # 保存识别结果
                    if last_ocr_text.strip():  # 确保存在有效文本
                        with open("ocr_results.txt", "a", encoding="utf-8") as f:
                            f.write(f"=== OCR结果 ===\n{last_ocr_text}\n\n")
                        print("识别结果已保存到 ocr_results.txt")
                    else:
                        print("未获取到有效的OCR结果。")
                    ocr_active = False  # 确保OCR激活状态正确关闭
                elif not keyboard.is_pressed('o'):
                    ocr_active = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        capturer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()