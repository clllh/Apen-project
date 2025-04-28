from camera.frame_capturer import VideoCapturer
from utils.fps_controller import FPSCounter, FPSLimiter
import cv2

def main():
    # 初始化
    capturer = VideoCapturer(src=0, target_fps=25)
    fps_counter = FPSCounter()
    fps_limiter = FPSLimiter(target_fps=25)
    
    try:
        while True:
            # 控制帧率
            fps_limiter.wait()
            
            # 捕获帧
            ret, prev_frame, curr_frame = capturer.read()
            if not ret:
                break
            
            # 更新FPS计数
            fps_counter.update()
            
            # 处理帧 (示例：显示帧差)
            if prev_frame is not None and curr_frame is not None:
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(gray_prev, gray_curr)
                cv2.imshow('Frame Difference', frame_diff)
            
            # 显示当前帧和FPS
            cv2.putText(curr_frame, f"FPS: {fps_counter.fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Live Feed', curr_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capturer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()