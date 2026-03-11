from paddleocr import PaddleOCR
import sys

print("Testing PaddleOCR initialization...")
try:
    # Try with use_gpu
    print("Attempt 1: use_gpu=True")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    print("Success with use_gpu=True")
except Exception as e:
    print(f"Attempt 1 failed: {e}")

try:
    # Try with device
    print("Attempt 2: device='gpu'")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', device='gpu')
    print("Success with device='gpu'")
except Exception as e:
    print(f"Attempt 2 failed: {e}")

try:
    # Try minimal
    print("Attempt 3: minimal")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    print("Success with minimal")
except Exception as e:
    print(f"Attempt 3 failed: {e}")
