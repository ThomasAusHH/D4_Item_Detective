from ultralytics import YOLO
from PIL import Image
from mss import mss
import cv2
import numpy as np
import time

def capture_screen(monitor_index=1):
    with mss() as sct:
        screenshot = sct.grab(sct.monitors[monitor_index])
    return Image.frombytes('RGB', screenshot.size, screenshot.rgb)



def detect_objects_on_image(image):
    model = YOLO("best.pt")
    results = model.predict(image)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

def draw_boxes_on_image(image, boxes):
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2, class_name, prob = box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image_np, f'{class_name}: {prob}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow('Object Detection', image_np)

if __name__ == "__main__":
    framerate = 2  # set the desired framerate here
    sleep_time = 1 / framerate
    monitor_index = 1  # set the index of the monitor to capture
    while True:
        screen = capture_screen(monitor_index)
        boxes = detect_objects_on_image(screen)
        draw_boxes_on_image(screen, boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(sleep_time)

