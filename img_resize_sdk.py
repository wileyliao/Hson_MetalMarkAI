from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(640, 640)):
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 獲取資料夾中所有檔案
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    file_number = 1

    # 遍歷每個檔案
    for file in files:
        # 構建完整的檔案路徑
        file_path = os.path.join(input_folder, file)

        try:
            # 打開圖片
            with Image.open(file_path) as img:
                # 調整圖片大小
                img = img.resize(size, Image.Resampling.LANCZOS)

                # 構建新的檔案名和路徑
                new_file_name = f"m{file_number:02}.jpg"
                new_file_path = os.path.join(output_folder, new_file_name)

                # 保存調整大小後的圖片
                img.save(new_file_path, "JPEG")
                print(file,"SAVED!!")

                # 更新檔案編號
                file_number += 1

        except Exception as e:
            print(f"Error processing {file}: {e}")


