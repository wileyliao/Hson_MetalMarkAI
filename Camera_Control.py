import os.path
from pypylon import pylon
import cv2

def generate_file_path_and_name(db_path, api_product_code, api_date_info, api_time_info):
    absolute_path_to_db = os.path.join(db_path, api_product_code, api_date_info)

    if not os.path.exists(absolute_path_to_db):
        os.makedirs(absolute_path_to_db)

    db_image_file_name = f'{api_time_info}.png'
    return absolute_path_to_db, db_image_file_name


class CameraHandler:
    def __init__(self):
        exposure_time = 4000.0
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.ExposureTime.SetValue(exposure_time)
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def capture_image(self):
        current_image_file = 'current_image.png'

        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(4000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = grab_result.Array

                cv2.imwrite(current_image_file, image)
                print("image saved")
                break
            grab_result.Release()
        print("image capture completed")
        return current_image_file

    def release(self):
        self.camera.StopGrabbing()
        self.camera.Close()

    def show_camera_feed(self):
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)  # 創建可調整大小的視窗

        # 預設座標值
        x, y, w, h = 506, 490, 2732, 1978

        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = grab_result.Array

                # 動態計算比例
                height, width = image.shape[:2]
                max_height = 600  # 最大高度
                scale = max_height / height if height > max_height else 1
                new_width = int(width * scale)
                new_height = int(height * scale)

                # 先縮放影像
                resized_image = cv2.resize(image, (new_width, new_height))

                # 縮小座標框
                scaled_x = int(x * scale)
                scaled_y = int(y * scale)
                scaled_w = int(w * scale)
                scaled_h = int(h * scale)

                # 在縮小後的影像上畫出矩形框
                cv2.rectangle(resized_image, (scaled_x, scaled_y), (scaled_x + scaled_w, scaled_y + scaled_h), (255, 255, 255), 2)

                # 顯示縮小後的影像
                cv2.imshow("Camera Feed", resized_image)

                # 檢查視窗是否被關閉
                if cv2.getWindowProperty("Camera Feed", cv2.WND_PROP_VISIBLE) < 1:
                    break

                # 檢查鍵盤輸入
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    # 按下 's' 鍵，保存原始影像中的該矩形區域
                    cropped_image = image[y:y + h, x:x + w]  # 截取原始影像中的該區域
                    save_path = 'cropped_image.png'
                    cv2.imwrite(save_path, cropped_image)
                    print(f"Image saved to {save_path}")
                elif key != 255:
                    break
            grab_result.Release()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    db_path = r'D:\test'
    absolute_path_to_db, image_file_name = generate_file_path_and_name(
        db_path,
        'test001',
        '20241004',
        '180000'
    )
    camera_handler = CameraHandler()
    camera_handler.show_camera_feed()
    camera_handler.release()  # 釋放相機
