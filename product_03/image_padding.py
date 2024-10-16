import cv2
import os


def pad_images_to_square(input_folder, output_folder, target_size=1280):
    """
    將 input_folder 中的所有影像填充至 target_size x target_size 的正方形，
    並將結果以相同名稱儲存至 output_folder。

    :param input_folder: 原始影像資料夾路徑
    :param output_folder: 處理後影像儲存的資料夾路徑
    :param target_size: 填充後的目標尺寸（預設為 3200）
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

            # 獲取影像的原始尺寸
            h, w = image.shape[:2]

            # 計算填充量
            pad_top = (target_size - h) // 2
            pad_bottom = target_size - h - pad_top
            pad_left = (target_size - w) // 2
            pad_right = target_size - w - pad_left

            # 使用黑色填充（或根據需求更改填充顏色）
            padded_image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            # 儲存處理後的影像
            cv2.imwrite(output_path, padded_image)
            print(f"已處理並儲存影像: {output_path}")


# 使用範例
input_folder = './data/local/origin'  # 替換成你的輸入資料夾路徑
output_folder = './data/local/padding'  # 替換成你的輸出資料夾路徑
pad_images_to_square(input_folder, output_folder)
