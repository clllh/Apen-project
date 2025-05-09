import cv2
import keyboard  # 需要安装：pip install keyboard
from pen_detector import PenDetector
from region_selector import RegionSelector

def main():
    cap = cv2.VideoCapture(0)
    detector = PenDetector()
    selector = RegionSelector()
    is_selecting = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测笔尖
        position, radius = detector.detect(frame)
        detector.draw_marker(frame, position, radius)
        
        # 按键控制
        if keyboard.is_pressed('space'):  # 按住空格开始选择
            if position and not is_selecting:
                selector.set_start_point(position)
                is_selecting = True
        else:
            if is_selecting and position:
                selector.set_end_point(position)
                is_selecting = False
                
                # 获取选定区域
                rect = selector.get_selection_rect()
                if rect:
                    x, y, w, h = rect
                    selected_region = frame[y:y+h, x:x+w]
                    cv2.imshow("Selected Text", selected_region)
                    # 这里可以添加OCR处理代码
        
        # 绘制选择区域
        if is_selecting and position:
            selector.set_end_point(position)
        selector.draw_selection(frame)
        
        # 显示界面
        cv2.imshow("Pen Reader Prototype", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()