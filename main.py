import cv2
import numpy as np
import matplotlib.pyplot as plt
from model_loader import load_yolo_model, load_ocr_model
from preprocess import crop_image, preprocess_for_ocr

MODEL_PATH = "model/best.pt"  
IMAGE_PATH = "Dataset/sample/HMCU_915451.jpg"

def extract_container_numbers():
    model = load_yolo_model(MODEL_PATH)
    ocr = load_ocr_model()

    results = model.predict(source=IMAGE_PATH, conf=0.25)
    final_extracted_text = ""

    for result in results:
        img = cv2.imread(result.path)
        img_with_boxes = img.copy()

        container_boxes = []

        # Collect YOLO boxes (container_number only)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label == "container_number":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                container_boxes.append({
                    "coords": (x1, y1, x2, y2),
                    "conf": conf,
                    "x1": x1
                })

        # Sort left→right
        container_boxes.sort(key=lambda b: b["x1"])

        # Process each bounding box
        for box_info in container_boxes:
            x1, y1, x2, y2 = box_info["coords"]
            conf = box_info["conf"]

            cropped = crop_image(img, (x1, y1, x2, y2))
            # preprocessed = preprocess_for_ocr(cropped)

            ocr_result = ocr.predict(cropped)

            extracted_text = ""
            if ocr_result and len(ocr_result) > 0:

                # New PaddleOCR dictionary format
                if isinstance(ocr_result[0], dict):
                    rec_texts = ocr_result[0].get('rec_texts', [])
                    rec_scores = ocr_result[0].get('rec_scores', [])

                    for text_part, text_conf in zip(rec_texts, rec_scores):
                        extracted_text += text_part.strip() + " "
                        print(f"  OCR_conf={text_conf:.4f}")

                # Older list format
                elif isinstance(ocr_result[0], list):
                    for line in ocr_result[0]:
                        text_part = line[1][0]
                        extracted_text += text_part.strip() + " "

            final_extracted_text += extracted_text.strip() + " "

            print(f"[container_number] {extracted_text.strip()}")

            # Draw bounding box + annotate text
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_with_boxes, extracted_text.strip(),
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

        # Show annotated output
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("YOLO + OCR Results")
        plt.show()

    print("\n✅ Final concatenated OCR text (left→right):")
    print(final_extracted_text.strip())


if __name__ == "__main__":
    extract_container_numbers()