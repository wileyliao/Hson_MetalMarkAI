import time
import threading
import tkinter as tk
from tkinter import ttk

def get_user_input():
    """
        Prompt the user to select the camera resolution and exposure settings.

        Returns:
        tuple: A tuple containing width (int), height (int), and exposure (float).
    """
    while True:
        print("Choose the camera resolution:")
        print("1. 4K (3840 x 2160)")
        print("2. 2K (2560 x 1440)")
        print("3. 1080p (1920 x 1080)")
        print("4. 720p (1280 x 720)")
        resolution_choice = input("Enter your choice (1, 2, 3, or 4): ")

        if resolution_choice == '1':
            width, height = 3840, 2160
            break
        elif resolution_choice == '2':
            width, height = 2560, 1440
            break
        elif resolution_choice == '3':
            width, height = 1920, 1080
            break
        elif resolution_choice == '4':
            width, height = 1280, 720
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    while True:
        exposure = input("Enter the camera exposure value (e.g., -6 for lower exposure, 0 for auto): ")
        try:
            exposure = float(exposure)
            break
        except ValueError:
            print("Invalid exposure value. Please enter a numeric value (e.g., -6, -4, 0).")

    return width, height, exposure

