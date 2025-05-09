from camera.frame_capturer import VideoCapturer
from utils.fps_controller import FPSCounter, FPSLimiter
from pen.detector import PenDetector
from utils.region_selector import RegionSelector
import cv2
import keyboard  # 需要安装：pip install keyboard

def main():
    # 初始化模块
    capturer = VideoCapturer(src=0, target_fps=25)
    fps_counter = FPSCounter()
    fps_limiter = FPSLimiter(target_fps=25)
    pen_detector = PenDetector()
    region_selector = RegionSelector()
    
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
            
            # 笔尖检测（仅在当前帧处理）
            pen_result = pen_detector.detect(curr_frame)
            if pen_result:
                pen_position, pen_radius = pen_result
            else:
                pen_position, pen_radius = None, None
            
            # 按键控制区域选择
            if keyboard.is_pressed('space'):
                if pen_position and not region_selector.is_selecting:
                    region_selector.begin_selection(pen_position)
            else:
                if region_selector.is_selecting and pen_position:
                    region_selector.end_selection()
            
            # 实时更新选择区域
            if region_selector.is_selecting and pen_position:
                region_selector.update_selection(pen_position)
            
            # 显示处理
            display_frame = curr_frame.copy()
            
            # 1. 显示笔尖位置
            if pen_position:
                cv2.circle(display_frame, pen_position, int(pen_radius), (0, 0, 255), 2)
                cv2.putText(display_frame, f"Pen: {pen_position}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 2. 显示选择区域
            region_selector.draw(display_frame)
            

            
            # 显示主画面
            cv2.putText(display_frame, f"FPS: {fps_counter.fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Pen Reader System', display_frame)
            
            # 显示选定区域（如果完成选择）
            if not region_selector.is_selecting and region_selector.get_region():
                x, y, w, h = region_selector.get_region()
                selected_roi = curr_frame[y:y+h, x:x+w]
                cv2.imshow("Selected Region", selected_roi)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        capturer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()