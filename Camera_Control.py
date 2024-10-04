import os.path
from pypylon import pylon
import cv2


class CameraHandler:
    def __init__(self):
        exposure_time = 4000.0
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.ExposureTime.SetValue(exposure_time)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def capture_image(self, absolute_path_to_db, image_file_name):
        current_image_file = 'current_image.png'

        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = grab_result.Array

                ssd_image_path = os.path.join(absolute_path_to_db, image_file_name)
                cv2.imwrite(current_image_file, image)
                cv2.imwrite(ssd_image_path, image)
                print("image saved")
                break
            grab_result.Release()
        print("image capture completed")
        return current_image_file

    def release(self):
        self.camera.StoptGrabbing()
        self.camera.Close()
