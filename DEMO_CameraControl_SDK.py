import cv2
import time

def countdown_to_close(cap):
    """
        Countdown function to close the camera and release resources.
        Args:
        cap (cv2.VideoCapture): Opened video capture object.
    """
    for i in range(2, 0, -1):
        print(f"Closing in {i} seconds...", end='\r')
        time.sleep(1)
    print("Closing in 0 seconds...", end='\r')
    cap.release()
    cv2.destroyAllWindows()
    exit()

def show_progress():
    """
        Simulate loading progress for opening the video frame.
    """
    print("AI model preparing...")
    for i in range(101):
        time.sleep(0.05)  # Simulate loading time
        print(f"AI model preparing... {i}%", end='\r')

    print("AI model preparing... 100%")
def initialize_camera(width, height, exposure):
    """
    Initialize the camera with the given width, height, and exposure settings.

    Args:
    width (int): Width of the video frame.
    height (int): Height of the video frame.
    exposure (float): Exposure setting for the camera.

    Returns:
    cv2.VideoCapture: Initialized video capture object.

    Raises:
    Exception: If the camera cannot be opened or settings cannot be applied.
    """

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Cannot open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    '''
    # Verify camera settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    print()
    if actual_width != width or actual_height != height or actual_exposure != exposure:
        cap.release()
        cv2.destroyAllWindows()
    raise Exception("Error: Camera settings could not be applied correctly")    
    '''

    print(f"Requested settings: width={width}, height={height}, exposure={exposure}")

    return cap


def calculate_fps(pTime):
    """
    Calculate the frames per second (FPS).

    Args:
    pTime (float): Previous time.

    Returns:
    tuple: Current time and FPS value.
    """
    cTime = time.perf_counter()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    return cTime, fps

def display_fps(frame, fps):
    """
    Display the FPS on the video frame.

    Args:
    frame (ndarray): The video frame.
    fps (float): The frames per second (FPS) value.
    """
    cv2.putText(frame, f'FPS:{int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

def display_status_text(frame, status_text):
    """
    Display the status text on the video frame.

    Args:
    frame (ndarray): The video frame.
    status_text (str): The status text to display.
    """
    cv2.putText(frame, status_text, (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


