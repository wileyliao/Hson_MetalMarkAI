from flask import Flask, request, jsonify
from ultralytics import YOLO
from Camera_Control import CameraHandler
from File_Management import generate_file_path_and_name
from product_01.product_01_main import product_01_main
from product_03.product_03_main import product_03_main
import json

app = Flask(__name__)
# camera_handler = CameraHandler()

db_path = r'C:\ichun_test'


with open('path_config.json', 'r') as file:
    path_config = json.load(file)


product_function_model_map = {
    'product_03': (
        product_03_main,
        YOLO(path_config["product_03_model_global_path"]),
        YOLO(path_config["product_03_model_local_path"]))
}


test_usage_image_path = path_config["product_03_test_image_path"]


@app.route('/MetalMarkAI', methods=['POST'])
def main():
    try:
        data = request.json['Data'][0]
        print("request received...")
        detect_stage = data.get('stage')

        # 取得矩陣大小
        product_matrix_row, product_matrix_column = map(int, data.get('matrix').split(','))
        # 取得產品料號

        product_code = data.get('product')
        date_info, time_info = data.get('op_time').split(' ')

        # absolute_path_to_db, image_file_name = generate_file_path_and_name(db_path, product_code, date_info, time_info)

        # current_image = camera_handler.capture_image(absolute_path_to_db, image_file_name)

        # test_image = r"C:\Projects\upload\MetalMarkAI\product_01\test888.png"
        product_detected_result = {}
        product_execution_time = 0

        if product_code in product_function_model_map:
            product_main_function, model_global, model_local = product_function_model_map[product_code]

            product_detected_result, product_execution_time = product_main_function(
                test_usage_image_path,
                model_global, model_local
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
                "ValueAry": value_ary,
                "timeTaken": f"{product_execution_time}"
            }
        )

    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(port=2486)

