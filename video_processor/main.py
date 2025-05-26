# from camera.frame_capturer import VideoCapturer
# from utils.fps_controller import FPSCounter, FPSLimiter
# from pen.detector import PenDetector
# from utils.region_selector import RegionSelector
# import cv2
# import keyboard  # 需要安装：pip install keyboard

# def main():
#     # 初始化模块
#     capturer = VideoCapturer(src=0, target_fps=25)
#     fps_counter = FPSCounter()
#     fps_limiter = FPSLimiter(target_fps=25)
#     pen_detector = PenDetector()
#     region_selector = RegionSelector()
    
#     try:
#         while True:
#             # 控制帧率
#             fps_limiter.wait()
            
#             # 捕获双帧
#             ret, prev_frame, curr_frame = capturer.read()
#             if not ret:
#                 break
            
#             # 更新FPS计数
#             fps_counter.update()
            
#             # 笔尖检测（仅在当前帧处理）
#             pen_result = pen_detector.detect(curr_frame)
#             if pen_result:
#                 pen_position, pen_radius = pen_result
#             else:
#                 pen_position, pen_radius = None, None
            
#             # 按键控制区域选择
#             if keyboard.is_pressed('space'):
#                 if pen_position and not region_selector.is_selecting:
#                     region_selector.begin_selection(pen_position)
#             else:
#                 if region_selector.is_selecting and pen_position:
#                     region_selector.end_selection()
            
#             # 实时更新选择区域
#             if region_selector.is_selecting and pen_position:
#                 region_selector.update_selection(pen_position)
            
#             # 显示处理
#             display_frame = curr_frame.copy()
            
#             # 1. 显示笔尖位置
#             if pen_position:
#                 cv2.circle(display_frame, pen_position, int(pen_radius), (0, 0, 255), 2)
#                 cv2.putText(display_frame, f"Pen: {pen_position}", 
#                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
#             # 2. 显示选择区域
#             region_selector.draw(display_frame)
            

            
#             # 显示主画面
#             cv2.putText(display_frame, f"FPS: {fps_counter.fps:.1f}", 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.imshow('Pen Reader System', display_frame)
            
#             # 显示选定区域（如果完成选择）
#             if not region_selector.is_selecting and region_selector.get_region():
#                 x, y, w, h = region_selector.get_region()
#                 selected_roi = curr_frame[y:y+h, x:x+w]
#                 cv2.imshow("Selected Region", selected_roi)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     finally:
#         capturer.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
import threading
import queue
import time
import cv2
import keyboard

from camera.frame_capturer import VideoCapturer
from utils.fps_controller import FPSCounter, FPSLimiter
from pen.detector import PenDetector
from utils.region_selector import RegionSelector


def detection_worker(detect_queue, result_queue, pen_detector):
    """
    独立线程运行，不断从detect_queue取待检测帧，检测后结果放入result_queue
    """
    while True:
        item = detect_queue.get()
        if item is None:
            break  # 收到退出信号

        frame, event_type = item
        result = pen_detector.detect(frame)
        result_queue.put((event_type, result))
        detect_queue.task_done()

def main():
    capturer = VideoCapturer(src=0, target_fps=30)
    fps_counter = FPSCounter()
    # fps_limiter = FPSLimiter(target_fps=25)  # 暂时注释，观察帧率变化
    pen_detectors = [PenDetector() for _ in range(2)]  # 多检测实例
    region_selector = RegionSelector()

    detect_queue = queue.Queue()
    result_queue = queue.Queue()

    detect_threads = []
    for i in range(2):
        t = threading.Thread(target=detection_worker,
                             args=(detect_queue, result_queue, pen_detectors[i]),
                             daemon=True)
        t.start()
        detect_threads.append(t)

    space_was_pressed = False
    press_pen_position = None
    release_pen_position = None
    frame_count = 0

    try:
        while True:
            # fps_limiter.wait()  # 取消限制，尽量快

            ret, prev_frame, curr_frame = capturer.read()
            if not ret:
                break

            fps_counter.update()
            frame_count += 1

            space_is_pressed = keyboard.is_pressed('space')

            if space_is_pressed and not space_was_pressed:
                detect_queue.put((curr_frame.copy(), "press"))

            if not space_is_pressed and space_was_pressed:
                detect_queue.put((curr_frame.copy(), "release"))

            space_was_pressed = space_is_pressed

            while not result_queue.empty():
                event_type, pen_result = result_queue.get()
                if pen_result:
                    if event_type == "press":
                        press_pen_position = pen_result[0]
                        print("空格按下，记录起点：", press_pen_position)
                    elif event_type == "release":
                        release_pen_position = pen_result[0]
                        print("空格抬起，记录终点：", release_pen_position)
                        if press_pen_position is not None and release_pen_position is not None:
                            region_selector.set_region_by_two_points(press_pen_position, release_pen_position)
                        else:
                            print("警告：起点或终点为空，无法设置区域")
                else:
                    if event_type == "release":
                        print("空格抬起，未检测到笔尖")

            display_frame = curr_frame.copy()
            region_selector.draw(display_frame)
            cv2.putText(display_frame, f"FPS: {fps_counter.fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Pen Reader System', display_frame)

            region = region_selector.get_region()
            if region:
                x, y, w, h = region
                if w > 0 and h > 0:
                    selected_roi = curr_frame[y:y+h, x:x+w]
                    cv2.imshow("Selected Region", selected_roi)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        for _ in range(2):
            detect_queue.put(None)
        for t in detect_threads:
            t.join()
        capturer.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()