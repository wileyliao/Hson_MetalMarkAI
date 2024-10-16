import cv2
import os


def image_padding(image, pad_param):
    image_height, image_width = image.shape[:2]
    pad_top = (pad_param - image_height) // 2
    pad_bottom = pad_param - image_height - pad_top
    pad_left = (pad_param - image_width) // 2
    pad_right = pad_param - image_width - pad_left

    image_padded = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return image_padded


def map_and_crop(results, image_pad, resized_size=640, original_size=3200):
    """
    將 YOLOv8 的結果從 image_resized 映射回 image_pad 並裁剪出物件區域，
    返回裁剪後的影像列表和對應的框座標。

    Args:
        results: YOLOv8 偵測結果
        image_pad: 填充後的原始影像（3200x3200）
        resized_size: 縮小後影像的邊長，默認為 640（對應 image_resized）
        original_size: 填充後的影像邊長，默認為 3200（對應 image_pad）

    Returns:
        cropped_images: List of cropped images
        boxes: List of bounding box coordinates corresponding to each cropped image
    """
    # 計算縮放比例
    scale_factor = original_size / resized_size
    cropped_images = []
    boxes = []

    # 獲取結果中的每個框的座標，進行映射和裁剪
    for idx, box in enumerate(results[0].boxes.xyxy):
        # 取得縮小後的座標（x1, y1, x2, y2）
        x1_resized, y1_resized, x2_resized, y2_resized = box

        # 映射回到原始填充影像中的座標
        x1_padded = int(x1_resized * scale_factor)
        y1_padded = int(y1_resized * scale_factor)
        x2_padded = int(x2_resized * scale_factor)
        y2_padded = int(y2_resized * scale_factor)

        # 裁剪出物體區域
        cropped_image = image_pad[y1_padded:y2_padded, x1_padded:x2_padded]
        cropped_images.append(cropped_image)

        # 保留原始填充影像中的座標作為位置資訊
        boxes.append((x1_padded, y1_padded, x2_padded, y2_padded))

    return cropped_images, boxes


def process_cropped_images(cropped_images, model_local, pad_param=1280, resized_size=640):
    """
    處理每一張裁剪後的影像，進行填充、縮放並使用本地模型進行推理。

    Args:
        cropped_images: 裁剪後的影像列表。
        model_local: YOLOv8 的本地模型，用於進行偵測。
        pad_param: 填充後的目標尺寸，默認為 1280。
        resized_size: 縮放後的影像尺寸，默認為 640。

    Returns:
        results_dict: 以 id 為鍵，推理結果為值的字典。
    """
    results_dict = {}

    for idx, cropped_image in enumerate(cropped_images):
        # Step 1: 將每張裁剪後的圖片填充到 1280x1280
        image_padded = image_padding(cropped_image, pad_param)

        # Step 2: 將填充後的圖片縮小到 640x640
        image_cropped_resized = cv2.resize(image_padded, (resized_size, resized_size))

        # Step 3: 使用本地模型進行偵測
        results = model_local(image_cropped_resized, conf=0.5)
        # results_show = results[0].plot()
        # cv2.imshow(f'result_{idx}', results_show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 將結果存入字典，以 idx 為鍵
        results_dict[idx] = results

    return results_dict
