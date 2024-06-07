import os.path
from image_processing import preprocess_image, map_boxes, cut_metal_from_boxes
from model_operations import load_model, detect_objects, classify_metal, classify_and_mark_images

def main():
    image_path = 'test888.png'
    model_path = "model_v1_gray.pt"
    output_dir = "cropped_images"
    resize_width, resize_height = 640, 640
    debug_mode = False
    threshold = 0.4
    rows, cols = 3, 4

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 加載模型
        model = load_model(model_path)

        # 圖像預處理
        ori_tensor, rez_ori_tensor = preprocess_image(
            image_path,
            resize_width,
            resize_height,
            debug_mode
        )

        # 檢測物體
        find_metal_boxes = detect_objects(
            model,
            rez_ori_tensor,
            debug_mode
        )

        # 映射框到原始圖像
        mapped_boxes_between_images = map_boxes(
            find_metal_boxes,
            ori_tensor,
            rez_ori_tensor,
            rows,
            cols,
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
            model,
            metals_position,
            rows,
            cols,
            threshold,
            classify_metal,
            debug_mode
        )

        # 顯示分類結果
        print("Filtered result: ", classification_results)

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
