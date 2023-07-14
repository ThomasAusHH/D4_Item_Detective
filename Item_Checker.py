from ultralytics import YOLO
from PIL import Image
from mss import mss
import cv2
import numpy as np
import time
import pytesseract
from pytesseract import Output

def capture_screen(monitor_index=1):
    with mss() as sct:
        screenshot = sct.grab(sct.monitors[monitor_index])
    return Image.frombytes('RGB', screenshot.size, screenshot.rgb)

def process_image_for_ocr(img):
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian blur to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Performing OTSU threshold
    _, thresholded = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded

def detect_objects_on_image(image, labels_to_include=None, prob_threshold=0.6):
    model = YOLO("best.pt")
    results = model.predict(image)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        label = result.names[class_id]
        prob = round(box.conf[0].item(), 2)
        if (labels_to_include is None or label in labels_to_include) and prob > prob_threshold:
            roi = image.crop((x1, y1, x2, y2))  # Extract region from image
            roi_np = np.array(roi)
            roi_np = cv2.cvtColor(roi_np, cv2.COLOR_RGB2BGR)
            processed_roi = process_image_for_ocr(roi_np)
            text = pytesseract.image_to_string(processed_roi, config='--psm 11', lang='eng', output_type=Output.STRING).replace('\n', ' ')
            output.append([x1, y1, x2, y2, label, prob, text])
    return output

def draw_boxes_on_image(image, boxes):
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2, class_name, prob, text = box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image_np, f'{class_name}: {prob}, OCR: {text}', (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print(f'Class: {class_name}, Probability: {prob}, OCR: {text}')
    cv2.imshow('Object Detection', image_np)

if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    framerate = 1  # set the desired framerate here
    sleep_time = 1 / framerate
    monitor_index = 1  # set the index of the monitor to capture
    while True:
        screen = capture_screen(monitor_index)
        boxes = detect_objects_on_image(screen, ["Item-Power", "Item-Affixes", "Item-Aspect"], prob_threshold=0.35)
        draw_boxes_on_image(screen, boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(sleep_time)
