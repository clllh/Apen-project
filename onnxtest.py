import onnxruntime as ort
import cv2
import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, ratio, dw, dh

# === 路径 ===
model_path = r"E:\aPenproject\Apen-project\video_processor\pen\best.onnx"
image_path = r"E:\aPenproject\Apen-project\testpic.jpg"

# === 加载模型 ===
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# === 加载图像并预处理 ===
img0 = cv2.imread(image_path)
img, ratio, dw, dh = letterbox(img0, new_shape=(640, 640))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img_rgb.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # HWC to CHW
img = np.expand_dims(img, axis=0)  # Add batch dimension

# === 推理 ===
outputs = session.run([output_name], {input_name: img})
detections = outputs[0][0]

# === 处理检测结果 ===
conf_thres = 0.25
for det in detections:
    conf = det[4]
    if conf < conf_thres:
        continue

    # 提取第一个关键点（假设是笔尖）
    x_kpt = det[5]
    y_kpt = det[6]
    c_kpt = det[7]

    if c_kpt < 0.2:
        continue

    # 去padding & 缩放还原
    kpt_x = int((x_kpt - dw) / ratio)
    kpt_y = int((y_kpt - dh) / ratio)

    print(f"笔尖关键点坐标：({kpt_x}, {kpt_y})，置信度：{c_kpt:.2f}")
    cv2.circle(img0, (kpt_x, kpt_y), 5, (0, 255, 0), -1)

# === 显示结果 ===
cv2.imshow("ONNX Pose Result", img0)
cv2.waitKey(0)
cv2.destroyAllWindows()