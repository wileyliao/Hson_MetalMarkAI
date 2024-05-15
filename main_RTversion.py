import cv2
from ultralytics import YOLO
import time


def check_inside(box1, box2):
    x1, y1, x2, y2 = box1
    x1i, y1i, x2i, y2i = box2
    return x1 <= x1i and x2 >= x2i and y1 <= y1i and y2 >= y2i

model = YOLO('model_v1.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

start_detection = False
status_text = "Press 's' to start, 'e' to end detection"
pTime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 640))

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}', (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        start_detection = True
        status_text = "Detection started.   Press 'e' to end, 'q' to exit."
    elif key == ord('e'):
        start_detection = False
        status_text = "Detection ended.   Press 's' to start, 'q' to exit."
    elif key == ord('q'):
        break
    cv2.putText(frame, status_text, (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if start_detection:
        results = model(frame)
        boxes_by_class = {'metal': [], 'pass': [], 'fail': []}
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, score, class_id = map(int, box)
                class_name = result.names[class_id]
                if class_name in boxes_by_class:
                    boxes_by_class[class_name].append((x1, y1, x2, y2))
                    color = (0, 0, 255) if class_name == 'fail' else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        inside_info = [
            (metal_box, label)
            for metal_box in boxes_by_class['metal']
            for label in ['pass', 'fail']
            for box in boxes_by_class[label]
            if check_inside(metal_box, box)
        ]

        for box, label in inside_info:
            mx1, my1, mx2, my2 = box
            center_x = (mx1 + mx2) // 2
            center_y = (my1 + my2) // 2
            text = label.upper()
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            start_x = center_x - text_size[0] // 2
            start_y = center_y + text_size[1] // 2
            text_color = (0, 0, 255) if label == 'fail' else (0, 255, 0)
            cv2.putText(frame, text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)
    cv2.imshow('Detection', frame)


cap.release()
cv2.destroyAllWindows()
