import os.path

from .product_01_image_processing import preprocess_image, map_boxes, cut_metal_from_boxes
from .product_01_model_operations import load_model, detect_objects, classify_and_mark_images
import time


def product_01_main(image_path, product_01_model, product_01_rows, product_01_columns):
    try:
        output_dir = "cropped_images"
        resize_width, resize_height = 640, 640
        debug_mode = False
        threshold = 0.4

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        start_time = time.time()

        # 圖像預處理
        ori_tensor, rez_ori_tensor = preprocess_image(
            image_path,
            resize_width,
            resize_height,
            debug_mode
        )

        # 檢測物體
        find_metal_boxes = detect_objects(
            product_01_model,
            rez_ori_tensor,
            debug_mode
        )

        # 映射到原始圖像並取得矩陣位置
        mapped_boxes_between_images = map_boxes(
            find_metal_boxes,
            ori_tensor,
            rez_ori_tensor,
            product_01_rows,
            product_01_columns,
            output_dir,
            debug_mode
        )

        # 裁剪和縮放物件
        metals_position = cut_metal_from_boxes(
            mapped_boxes_between_images,
            ori_tensor,
            resize_width,
            resize_height,
            debug_mode
        )

        # 分類和標記
        classification_results = classify_and_mark_images(
            product_01_model,
            metals_position,
            threshold,
            debug_mode
        )
        end_time = time.time()
        execution_time = end_time - start_time

        # show_result =

        return classification_results, execution_time

    except Exception as e:
        raise e


if __name__ == "__main__":
    model_path = 'model_v1_gray.pt'
    model = load_model(model_path)
    result, time = product_01_main(model, 3, 4)

    print('....result....')
    print(result)

    print('....time....')
    print(time)
