import glob
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR

def load_yolo_model(model_path):
    best_model_path = glob.glob(model_path)[0]
    print(f"[INFO] Loading YOLO model: {best_model_path}")
    model = YOLO(best_model_path)
    return model

def load_ocr_model():
    print("[INFO] Loading OCR model...")
    return PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,use_textline_orientation=False)