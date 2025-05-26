import cv2
import pytesseract
from PIL import Image

class OCRProcessor:
    def __init__(self):
        # 如果 pytesseract 无法自动找到 Tesseract OCR 引擎，自行指定路径
        # pytesseract.pytesseract.tesseract_cmd = r'<Tesseract OCR 安装路径>\tesseract.exe'
        self.keywords = ["example", "test", "demo"]  # 预设的关键词列表

    def process(self, image):
        try:
            # 将 ROI 转换为 PIL 图像格式
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # 进行 OCR 识别
            ocr_result = pytesseract.image_to_string(pil_image)
            return ocr_result
        except Exception as e:
            print(f"OCR processing error: {e}")
            return ""

    def filter_text(self, text):
        # 示例：移除特殊字符并转换为小写
        filtered_text = ''.join(e for e in text if e.isalnum() or e.isspace()).lower()
        return filtered_text

    def compare_with_keywords(self, text):
        filtered_text = self.filter_text(text)
        for keyword in self.keywords:
            if keyword in filtered_text:
                return True, keyword
        return False, None