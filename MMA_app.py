from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import os
from Camera_Control import camera_handler
from File_Management import generate_file_path_and_name

from product_01.product_01_main import product_01_main

app = Flask(__name__)
# camera_handler = CameraHandler()
db_path = r'C:\ichun_test'


def load_model(model_path):
    model = YOLO(model_path)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model


product_01_model_path = "C:/AI_detection/Hson_MetalMarkAI/project_01/model_v1_gray.pt"
product_01_model = load_model(product_01_model_path)

product_function_and_model_map = {
    'product_01': (product_01_main, product_01_model)
}


@app.route('/MetalMarkAI', methods=['POST'])
def main():
    try:
        data = request.json['Data'][0]
        print("request received...")
        detect_stage = data.get('stage')
        product_matrix_row, product_matrix_column = map(int, data.get('matrix').split(','))
        product_code = data.get('product')
        date_info, time_info = data.get('op_time').split(' ')

        absolute_path_to_db, image_file_name = generate_file_path_and_name(db_path, product_code, date_info, time_info)

        print(f"Save image as name: {image_file_name}, in folder: {absolute_path_to_db}")

        # current_image = camera_handler.capture_image(absolute_path_to_db, image_file_name)
        # print(current_image)
        test_image = "C:/AI_detection/Hson_MetalMarkAI/project_01/test888.png"

        if product_code in product_function_and_model_map:
            product_main_function, product_model = product_function_and_model_map[product_code]
            product_detected_result, product_execution_time = product_main_function(
                test_image,
                product_model,
                product_matrix_row,
                product_matrix_column
            )

            # 提取fail座標
            fail_coords = [f"{key[0]}, {key[1]}" for key, value in product_detected_result.items() if value == 'fail']

            # 判斷是否有fail的結果
            value_ary = fail_coords if fail_coords else "pass"



            return jsonify(
                {
                    "Data": [
                        {
                            "stage": f"{detect_stage}",
                            "op_time": f"{date_info} {time_info}",
                            "product": f"{product_code}",
                            "details": f"{product_detected_result}"
                        }
                    ],
                    "ValueAry": f"{value_ary}",
                    "timeTaken": f"{product_execution_time}"
                }
            )

    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(port=2486)
    camera_handler.release()
