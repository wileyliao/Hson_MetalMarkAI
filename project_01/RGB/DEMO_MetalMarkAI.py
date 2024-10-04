import cv2
import time
from ultralytics import YOLO
import threading
import logging
from DEMO_CameraControl_SDK import (
    countdown_to_close, initialize_camera, calculate_fps,
    display_fps, display_status_text, show_progress
)
from DEMO_System_Setting_SDK import (
    get_user_input,
    gpu_get_stats
)
from DEMO_AI_Detection_SDK import process_frame
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
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
    progress_thread.join()
    print("Press 's' to start detection.")
    start_detection = False
    pause_detection = False
    status_text = "Press 's' to start detection or 'p' to pause"
    pTime = 0
    confidence_threshold = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        pTime, fps = calculate_fps(pTime)
        display_fps(frame, fps)
        display_status_text(frame, status_text)


        if start_detection and not pause_detection:
            boxes_by_class = process_frame(frame, model, confidence_threshold)

            for idx, metal_box in enumerate(boxes_by_class['metal']):
                # Crop and resize metal box
                mx1, my1, mx2, my2 = metal_box
                metal_crop = frame[my1:my2, mx1:mx2]
                #resized_metal_crop = cv2.resize(metal_crop, (640, 640), interpolation=cv2.INTER_AREA)

                # Detect pass/fail inside the cropped and resized metal box
                resized_results = model(metal_crop)
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
        logging.debug(gpu_get_stats())
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

if __name__ == "__main__":
    main()