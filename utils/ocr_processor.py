from paddleocr import PaddleOCR
import cv2
import numpy as np

class OCRProcessor:
    def __init__(self):
        # 初始化 PaddleOCR，使用默认的中文模型
        self.ocr = PaddleOCR(lang="ch")

    def process_image(self, image):
        """
        对图像进行文字识别
        :param image: 输入图像（numpy 数组）
        :return: 识别结果，包括文字框坐标、文字内容和置信度
        """
        if image is None:
            print("图像为空")
            return None, None, None

        # 进行文字检测和识别
        results = self.ocr.ocr(image)

        # 解析结果
        boxes = []
        texts = []
        scores = []

        for line in results[0]:
            if isinstance(line, list) and len(line) >= 2:
                box = line[0]
                text_info = line[1]
                if isinstance(text_info, dict) and "text" in text_info and "score" in text_info:
                    text = text_info["text"]
                    score = text_info["score"]
                    boxes.append(box)
                    texts.append(text)
                    scores.append(score)
                else:
                    print(f"Unexpected text_info format: {text_info}")
            else:
                print(f"Unexpected line format: {line}")

        return boxes, texts, scores

    def draw_results(self, image, boxes, texts, scores):
        """
        在图像上绘制文字识别结果
        :param image: 输入图像（numpy 数组）
        :param boxes: 文字框坐标
        :param texts: 识别的文字内容
        :param scores: 识别的置信度
        :return: 绘制结果的图像
        """
        for box, text, score in zip(boxes, texts, scores):
            # 绘制文字框
            box = np.array(box).astype(np.int32)
            cv2.polylines(image, [box], True, (0, 255, 0), 2)
            # 绘制文字
            cv2.putText(image, f"{text} ({score:.2f})", (box[0][0], box[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image