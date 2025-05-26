from ultralytics import YOLO
import cv2

# 1. 加载 YOLOv8 Pose 模型（.pt 文件）
model_path = r"E:\aPenproject\Apen-project\video_processor\pen\bestva.pt"
model = YOLO(model_path)

# 2. 读取测试图像
img_path = r"E:\aPenproject\Apen-project\test3.jpg"
image = cv2.imread(img_path)

# 3. 推理
results = model.predict(source=image, conf=0.25, save=False, verbose=True)

# 4. 提取关键点
for result in results:
    keypoints = result.keypoints  # N × K × 3 (N个目标，K个关键点，每个点[x, y, conf])

    if keypoints is not None:
        for kp in keypoints.data:
            # kp 是 (2, 3)，表示2个关键点 (tip, tail)，每个包含 (x, y, conf)
            tip_x, tip_y, tip_conf = kp[0]
            tail_x, tail_y, tail_conf = kp[1]

            # 只在置信度高于阈值时可视化
            if tip_conf > 0.2:
                print(f"笔尖(Tip): ({tip_x.item():.1f}, {tip_y.item():.1f}), 置信度: {tip_conf.item():.2f}")
                cv2.circle(image, (int(tip_x.item()), int(tip_y.item())), 6, (0, 255, 0), -1)

            if tail_conf > 0.2:
                print(f"笔尾(Tail): ({tail_x.item():.1f}, {tail_y.item():.1f}), 置信度: {tail_conf.item():.2f}")
                cv2.circle(image, (int(tail_x.item()), int(tail_y.item())), 6, (0, 0, 255), -1)

# 5. 显示或保存图像
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
