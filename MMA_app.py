from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch

from Camera_Control import CameraHandler
from File_Management import generate_file_path_and_name

from product_01.product_01_main import product_01_main
from product_03.product_03_main import product_03_main

app = Flask(__name__)
# camera_handler = CameraHandler()
db_path = r'C:\ichun_test'


def load_model(model_path):
    model = YOLO(model_path)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model


product_01_model_path = r'product_01\model_v1_gray.pt'
product_01_model = load_model(product_01_model_path)

product_03_model_path = None
product_03_model = None

product_function_and_model_map = {
    'product_01': (product_01_main, product_01_model),
    'product_03': (product_03_main, product_03_model)
}


@app.route('/MetalMarkAI', methods=['POST'])
def main():
    try:
        "取得api內容"
        data = request.json['Data'][0]
        # 測試階段
        detect_stage = data.get('stage')
        # 取得矩陣大小
        product_matrix_row, product_matrix_column = map(lambda x: int(x) - 1, data.get('matrix').split(','))        # 取得產品料號
        product_code = data.get('product')
        # 取得操作日期、時間
        date_info, time_info = data.get('op_time').split(' ')

        absolute_path_to_db, image_file_name = generate_file_path_and_name(db_path, product_code, date_info, time_info)

        print(f"Save image as name: {image_file_name}, in folder: {absolute_path_to_db}")

        # current_image = camera_handler.capture_image(absolute_path_to_db, image_file_name)
        test_image = r'.\project_01\test888.png'

        if product_code in product_function_and_model_map:
            product_main_function, product_model = product_function_and_model_map[product_code]
            product_detected_result, product_execution_time = product_main_function(
                test_image,
                product_model,
                product_matrix_row,
                product_matrix_column
            )

            print(product_detected_result)

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
    # camera_handler.release()
