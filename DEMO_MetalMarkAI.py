import cv2
import time
from ultralytics import YOLO
import threading
from DEMO_CameraControl_SDK import (
    countdown_to_close, initialize_camera
)
from DEMO_System_Setting_SDK import (
    get_user_input, show_progress
)
from DEMO_AI_Detection_SDK import (
    check_inside, process_frame
)

model = YOLO('model_v1.pt')
width, height, exposure = get_user_input()
print(f"Current settings:\nResolution: {width} x {height}\nExposure: {exposure}")

# Start a thread to show the progress of opening the video frame
progress_thread = threading.Thread(target=show_progress)
progress_thread.start()

try:
    cap = initialize_camera(width, height, exposure)
except Exception as e:
    print(e)
    exit()

# Wait for the progress thread to finish
progress_thread.join()
print("Press 's' to start detection.")
start_detection = False
pause_detection = False
status_text = "Press 's' to start detection or 'p' to pause"
pTime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    cTime = time.perf_counter()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    cv2.putText(frame, status_text, (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if start_detection and not pause_detection:
        boxes_by_class = process_frame(frame, model)

        for idx, metal_box in enumerate(boxes_by_class['metal']):
            # Crop and resize metal box
            mx1, my1, mx2, my2 = metal_box
            metal_crop = frame[my1:my2, mx1:mx2]
            resized_metal_crop = cv2.resize(metal_crop, (640, 640), interpolation=cv2.INTER_AREA)

            # Detect pass/fail inside the cropped and resized metal box
            resized_results = model(resized_metal_crop)
            detection_label = 'UNKNOWN'
            for result in resized_results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, score, class_id = map(int, box)
                    class_name = result.names[class_id]
                    if class_name in ['pass', 'fail']:
                        detection_label = class_name.upper()
                        break

            # Add label to the metal box
            text_size = cv2.getTextSize(detection_label, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            center_x = (mx1 + mx2) // 2
            center_y = (my1 + my2) // 2
            start_x = center_x - text_size[0] // 2
            start_y = center_y + text_size[1] // 2
            text_color = (0, 0, 255) if detection_label == 'FAIL' else (0, 255, 0)
            cv2.putText(frame, detection_label, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)

    cv2.imshow('Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        start_detection = True
        pause_detection = False
        status_text = "Detection in progress... Press 'p' to pause"
        print("Detection started.")
    elif key == ord('p'):
        pause_detection = True
        start_detection = False
        status_text = "Detection paused. Press 's' to start"
        print("Detection paused.")
    elif key == ord('q'):
        print("Exiting...")
        countdown_to_close(cap)

cap.release()
cv2.destroyAllWindows()
