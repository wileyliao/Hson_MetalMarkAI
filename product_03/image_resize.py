import cv2
import os


def resize_images(input_folder, output_folder, target_size=640):
    """
    將 input_folder 中的所有影像縮放至 target_size x target_size，
    並以相同名稱儲存至 output_folder。

    :param input_folder: 原始影像資料夾路徑
    :param output_folder: 縮放後影像儲存的資料夾路徑
    :param target_size: 縮放後的目標尺寸（預設為 640）
    """
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍歷輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        # 構建完整的檔案路徑
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 只處理圖像檔案
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 讀取影像
            image = cv2.imread(input_path)
            if image is None:
                print(f"無法讀取影像: {input_path}")
                continue

            # 縮放影像到目標尺寸
            resized_image = cv2.resize(image, (target_size, target_size))

            # 儲存處理後的影像
            cv2.imwrite(output_path, resized_image)
            print(f"已處理並儲存影像: {output_path}")


# 使用範例
input_folder = './data/padding'  # 替換成你的輸入資料夾路徑
output_folder = './data/resized'  # 替換成你的輸出資料夾路徑
resize_images(input_folder, output_folder)
