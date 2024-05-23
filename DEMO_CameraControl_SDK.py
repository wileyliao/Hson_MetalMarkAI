import cv2
import time

def countdown_to_close(cap):
    for i in range(2, 0, -1):
        print(f"Closing in {i} seconds...", end='\r')
        time.sleep(1)
    print("Closing in 0 seconds...", end='\r')
    cap.release()
    cv2.destroyAllWindows()
    exit()

def initialize_camera(width, height, exposure):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Cannot open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    return cap