import cv2
import numpy as np

def crop_image(img, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]

def preprocess_for_ocr(cropped):
    # Basic cleaning for OCR improvement
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 39, 10
    )
    return th