import os.path
import json
from flask import Flask, request, jsonify
from image_processing import preprocess_image, map_boxes, cut_metal_from_boxes
from model_operations import load_model, detect_objects, classify_metal, classify_and_mark_images

#app = Flask(__name__)
#@app.route('/MetalMarkAI', methods=['POST'])
def MetalMarkAI():
    try:
        # # 獲取 JSON 請求
        # data = request.get_json()
        #
        # detection = data.get('detection')
        # model_path = data.get('model')
        # image_path = data.get('filepath')
        #
        # # 驗證輸入
        # if detection != 'start' or not model_path or not image_path:
        #     return jsonify({"error": "Invalid input parameters"}), 400
        #
        # # 檢查文件是否存在
        # if not os.path.exists(model_path):
        #     return jsonify({"error": f"Model file {model_path} not found"}), 404
        #
        # if not os.path.exists(image_path):
        #     return jsonify({"error": f"Image file {image_path} not found"}), 404

        image_path = 'test888.png'
        model_path = "model_v1_gray.pt"

        output_dir = "cropped_images"
        resize_width, resize_height = 640, 640
        debug_mode = True
        threshold = 0.4
        rows, cols = 3, 4

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


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
        json_result = {
            "result": {
                "matrix_labels": {
                    f"{row},{col}": result if isinstance(result, str) else result["classification"]
                    for (row, col), result in classification_results.items()
                }
            }
        }
        # 顯示 JSON 格式的結果
        json_output = json.dumps(json_result, indent=4)
        print("Filtered result: ", json_output)
        #return jsonify(json_result), 200

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        #return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    MetalMarkAI()
    #app.run(host='0.0.0.0', port=5000)
