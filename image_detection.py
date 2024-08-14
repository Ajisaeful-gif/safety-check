import os
import cv2
import math
from ultralytics import YOLO
import io
import base64
from PIL import Image
import time

# Definisi kelas APD yang diperlukan
apd_required = {
    "Glasses": False,
    "Gloves": False,
    "Mask": False,
}

def process_image(img):
    model = YOLO("YOLO-Weights/best_model_deteksi.pt")
    classNames = ['Glasses', 'Gloves', 'Helmet', 'Mask', 'Safety Vest']

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

            color = None

            if class_name == 'Glasses' and conf > 0.25:
                color = (0, 204, 255)
            elif class_name == 'Gloves' and conf > 0.25:
                color = (222, 82, 175)
            elif class_name == 'Mask' and conf > 0.25:
                color = (255, 255, 0)

            if color:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(img, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            if class_name in apd_required and conf > 0.25:
                apd_required[class_name] = True

    apd_used_count = sum(apd_required.values())

    if apd_used_count == 3:  # Check if all 3 APD items are used
        cv2.putText(img, "APD sesuai", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(img, "APD tidak sesuai", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img