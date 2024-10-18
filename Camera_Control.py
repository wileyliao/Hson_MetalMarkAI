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

    def capture_image(self,temporary_folder, stage, abs_path_to_db, image_file_name):
        current_image_path = None
        while self.camera.IsGrabbing():

            grab_result = self.camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
            try:
                if grab_result.GrabSucceeded():
                    image = grab_result.Array
                    x, y, w, h = 506, 490, 2732, 1978
                    cropped_image = image[y:y + h, x:x + w]
                    image_rotate = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)

                    # 檢查影像是否為空
                    if image is not None and image.size > 0:
                        current_image_file = f'stage_0{stage}.png'
                        db_image_file = f'{image_file_name}_stage_0{stage}.png'

                        current_image_path = os.path.join(temporary_folder, current_image_file)
                        db_image_path = os.path.join(abs_path_to_db, db_image_file)

                        # 確保資料夾存在
                        os.makedirs(abs_path_to_db, exist_ok=True)
                        os.makedirs(temporary_folder, exist_ok=True)


                        cv2.imwrite(current_image_path, image_rotate)
                        cv2.imwrite(db_image_path, image_rotate)
                        print("image saved")
                        break
                    else:
                        print("Captured image is empty, skipping save.")
            finally:
                grab_result.Release()
        print("image capture completed")
        return current_image_path


    def release(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
            print("Stopped grabbing images.")
        if self.camera.IsOpen():
            self.camera.Close()
            print("Camera closed.")
        print("Camera resources released.")

