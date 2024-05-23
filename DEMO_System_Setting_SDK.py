import time

def get_user_input():
    while True:
        print("Choose the camera resolution:")
        print("1. 4K (3840 x 2160)")
        print("2. 2K (2048 x 1080)")
        resolution_choice = input("Enter your choice (1 or 2): ")

        if resolution_choice == '1':
            width, height = 3840, 2160
            break
        elif resolution_choice == '2':
            width, height = 2048, 1080
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    while True:
        exposure = input("Enter the camera exposure value (e.g., -6 for lower exposure, 0 for auto): ")
        try:
            exposure = float(exposure)
            break
        except ValueError:
            print("Invalid exposure value. Please enter a numeric value (e.g., -6, -4, 0).")

    return width, height, exposure

def show_progress():
    for i in range(101):
        time.sleep(0.05)  # Simulate loading time
        print(f"Opening video frame... {i}%", end='\r')
    print("Opening video frame... 100%")