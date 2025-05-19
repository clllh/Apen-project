import cv2
import easyocr

class OCRProcessor:
    def __init__(self, lang='ch_sim'):
        # 初始化 EasyOCR 识别器
        self.reader = easyocr.Reader([lang])  # 指定识别的语言（中文）

    def process(self, image):
        try:
            # 执行 OCR 识别
            results = self.reader.readtext(image, detail=0)
            # 提取识别结果
            ocr_text = '\n'.join(results)
            return ocr_text
        except Exception as e:
            print(f"OCR processing error: {e}")
            return ""