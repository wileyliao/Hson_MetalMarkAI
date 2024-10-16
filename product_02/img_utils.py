import cv2
import os
import base64

import numpy as np



def binary_img_check(img, start, end):
    """
    根據閥值範圍調整影像二值結果，並顯示圖片
    param
        img:圖片
        start: 起始threshold
        end: 終點threshold
    return:
        顯示每一次調整threshold圖片
    """
    for i, threshold_value in enumerate(range(start, end, 5), 1):
        _, img_bin = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
        cv2.imshow(f'{threshold_value}', img_bin)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# #檢查閥值該設定多少
# image_path = 'test.jpg'
# img = cv2.imread(image_path)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# init, final = 140, 240
# binary_img_check(img_gray, init,final)

def roi_check(folder_path, box_coords):
    """
    檢查roi在每張圖片標記的結果
    param:
        folder_path: 輸入資料夾路徑
        box_coords: 座標[(x1, y1), (x2, y2)]，表示 Box 的左上和右下角
    return:
        顯示每一張標記好box的圖片
    """

    box_coords = sorted(box_coords)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        # 在圖上標記 box1
        cv2.rectangle(image, box_coords[0], box_coords[1], (0, 255, 0), 2)
        cv2.imshow('Image with Boxes', image)

        # 按下 'q' 退出
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


# box_coords = [(1312, 684), (1617, 828)]
# folder_path = './test'
# roi_check(folder_path, box_coords)

def roi_save(folder_path,output_folder_box, box_coords):

    box_coords = sorted(box_coords)


    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        x1, y1 = box_coords[0]
        x2, y2 = box_coords[1]
        box1_crop = image[y1:y2, x1:x2]
        box1_save_path = os.path.join(output_folder_box, f"{os.path.splitext(image_file)[0]}_box1.jpg")
        cv2.imwrite(box1_save_path, box1_crop)


box_coords = [(2496, 1080), (2727, 1281)]
folder_path = './test'
# #建立資料存檔路徑
output_folder_box = 'box_coord'

os.makedirs(output_folder_box, exist_ok=True)
roi_save(folder_path, output_folder_box, box_coords)
