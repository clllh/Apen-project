# # from camera.frame_capturer import VideoCapturer
# # from utils.fps_controller import FPSCounter, FPSLimiter
# # from pen.detector import PenDetector
# # from utils.region_selector import RegionSelector
# # import cv2
# # import keyboard  # 需要安装：pip install keyboard

# # def main():
# #     # 初始化模块
# #     capturer = VideoCapturer(src=0, target_fps=25)
# #     fps_counter = FPSCounter()
# #     fps_limiter = FPSLimiter(target_fps=25)
# #     pen_detector = PenDetector()
# #     region_selector = RegionSelector()
    
# #     try:
# #         while True:
# #             # 控制帧率
# #             fps_limiter.wait()
            
# #             # 捕获双帧
# #             ret, prev_frame, curr_frame = capturer.read()
# #             if not ret:
# #                 break
            
# #             # 更新FPS计数
# #             fps_counter.update()
            
# #             # 笔尖检测（仅在当前帧处理）
# #             pen_result = pen_detector.detect(curr_frame)
# #             if pen_result:
# #                 pen_position, pen_radius = pen_result
# #             else:
# #                 pen_position, pen_radius = None, None
            
# #             # 按键控制区域选择
# #             if keyboard.is_pressed('space'):
# #                 if pen_position and not region_selector.is_selecting:
# #                     region_selector.begin_selection(pen_position)
# #             else:
# #                 if region_selector.is_selecting and pen_position:
# #                     region_selector.end_selection()
            
# #             # 实时更新选择区域
# #             if region_selector.is_selecting and pen_position:
# #                 region_selector.update_selection(pen_position)
            
# #             # 显示处理
# #             display_frame = curr_frame.copy()
            
# #             # 1. 显示笔尖位置
# #             if pen_position:
# #                 cv2.circle(display_frame, pen_position, int(pen_radius), (0, 0, 255), 2)
# #                 cv2.putText(display_frame, f"Pen: {pen_position}", 
# #                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
# #             # 2. 显示选择区域
# #             region_selector.draw(display_frame)
            

            
# #             # 显示主画面
# #             cv2.putText(display_frame, f"FPS: {fps_counter.fps:.1f}", 
# #                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# #             cv2.imshow('Pen Reader System', display_frame)
            
# #             # 显示选定区域（如果完成选择）
# #             if not region_selector.is_selecting and region_selector.get_region():
# #                 x, y, w, h = region_selector.get_region()
# #                 selected_roi = curr_frame[y:y+h, x:x+w]
# #                 cv2.imshow("Selected Region", selected_roi)
            
# #             if cv2.waitKey(1) & 0xFF == ord('q'):
# #                 break
                
# #     finally:
# #         capturer.release()
# #         cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     main()
# import threading
# import queue
# import time
# import cv2
# import keyboard

# from camera.frame_capturer import VideoCapturer
# from utils.fps_controller import FPSCounter, FPSLimiter
# from pen.detector import PenDetector
# from utils.region_selector import RegionSelector


# def detection_worker(detect_queue, result_queue, pen_detector):
#     """
#     独立线程运行，不断从detect_queue取待检测帧，检测后结果放入result_queue
#     """
#     while True:
#         item = detect_queue.get()
#         if item is None:
#             break  # 收到退出信号

#         frame, event_type = item
#         result = pen_detector.detect(frame)
#         result_queue.put((event_type, result))
#         detect_queue.task_done()

# def main():
#     capturer = VideoCapturer(src=0, target_fps=30)
#     fps_counter = FPSCounter()
#     # fps_limiter = FPSLimiter(target_fps=25)  # 暂时注释，观察帧率变化
#     pen_detectors = [PenDetector() for _ in range(2)]  # 多检测实例
#     region_selector = RegionSelector()

#     detect_queue = queue.Queue()
#     result_queue = queue.Queue()

#     detect_threads = []
#     for i in range(2):
#         t = threading.Thread(target=detection_worker,
#                              args=(detect_queue, result_queue, pen_detectors[i]),
#                              daemon=True)
#         t.start()
#         detect_threads.append(t)

#     space_was_pressed = False
#     press_pen_position = None
#     release_pen_position = None
#     frame_count = 0

#     try:
#         while True:
#             # fps_limiter.wait()  # 取消限制，尽量快

#             ret, prev_frame, curr_frame = capturer.read()
#             if not ret:
#                 break

#             fps_counter.update()
#             frame_count += 1

#             space_is_pressed = keyboard.is_pressed('space')

#             if space_is_pressed and not space_was_pressed:
#                 detect_queue.put((curr_frame.copy(), "press"))

#             if not space_is_pressed and space_was_pressed:
#                 detect_queue.put((curr_frame.copy(), "release"))

#             space_was_pressed = space_is_pressed

#             while not result_queue.empty():
#                 event_type, pen_result = result_queue.get()
#                 if pen_result:
#                     if event_type == "press":
#                         press_pen_position = pen_result[0]
#                         print("空格按下，记录起点：", press_pen_position)
#                     elif event_type == "release":
#                         release_pen_position = pen_result[0]
#                         print("空格抬起，记录终点：", release_pen_position)
#                         if press_pen_position is not None and release_pen_position is not None:
#                             region_selector.set_region_by_two_points(press_pen_position, release_pen_position)
#                         else:
#                             print("警告：起点或终点为空，无法设置区域")
#                 else:
#                     if event_type == "release":
#                         print("空格抬起，未检测到笔尖")

#             display_frame = curr_frame.copy()
#             region_selector.draw(display_frame)
#             cv2.putText(display_frame, f"FPS: {fps_counter.fps:.1f}",
#                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.imshow('Pen Reader System', display_frame)

#             region = region_selector.get_region()
#             if region:
#                 x, y, w, h = region
#                 if w > 0 and h > 0:
#                     selected_roi = curr_frame[y:y+h, x:x+w]
#                     cv2.imshow("Selected Region", selected_roi)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         for _ in range(2):
#             detect_queue.put(None)
#         for t in detect_threads:
#             t.join()
#         capturer.release()
#         cv2.destroyAllWindows()
# if __name__ == "__main__":
#     main()

# import cv2
# from utils.fps_controller import FPSCounter, FPSLimiter
# import time

# def test_camera_fps(camera_index=0, duration=15):
#     cap = cv2.VideoCapture(camera_index)

#     if not cap.isOpened():
#         print("无法打开摄像头")
#         return

#     # 设置分辨率
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

#     print("开始测试摄像头帧率...")

#     fps_counter = FPSCounter()
#     # fps_limiter = FPSLimiter(target_fps=30)  # 可以注释掉，测试摄像头最大帧率

#     frame_count = 0
#     start_time = time.time()

#     while True:
#         # fps_limiter.wait()  # 测试最高帧率时注释掉限制

#         ret, frame = cap.read()
#         if not ret:
#             print("读取帧失败")
#             break

#         fps_counter.update()

#         frame_count += 1
#         elapsed_time = time.time() - start_time

#         # 直接画在画面上
#         cv2.putText(frame, f"FPS: {fps_counter.fps:.2f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow("Camera FPS Test", frame)

#         if elapsed_time > duration or cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     total_time = time.time() - start_time
#     print(f"总帧数: {frame_count}")
#     print(f"总耗时: {total_time:.2f} 秒")
#     print(f"平均 FPS: {frame_count / total_time:.2f}")

# if __name__ == "__main__":
#     test_camera_fps(camera_index=0, duration=15)
import cv2
from utils.fps_controller import FPSCounter
import time
from ultralytics import YOLO
import numpy as np
import keyboard  # pip install keyboard

class PenDetector:
    def __init__(self, model_path=r'E:\aPenproject\Apen-project\video_processor\pen\bestva.pt', conf_thres=0.25):
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

def test_camera_with_pen_detection(camera_index=0, duration=60):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("开始测试摄像头并进行按空格框选笔尖...")

    fps_counter = FPSCounter()
    detector = PenDetector()

    frame_count = 0
    start_time = time.time()

    space_was_pressed = False
    press_pos = None
    release_pos = None
    region = None

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
                else:
                    print("起点或终点无效，无法设置区域")

        space_was_pressed = space_is_pressed

        # 画选中区域框
        if region is not None:
            x, y, w, h = region
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示FPS
        cv2.putText(frame, f"FPS: {fps_counter.fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera with Pen Detection and Region Selection", frame)

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > duration or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera_with_pen_detection(camera_index=0, duration=60)
