from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import os
from Camera_Control import CameraHandler
from File_Management import generate_file_path_and_name
import time
import random
from product_01.product_01_main import product_01_main

app = Flask(__name__)
# camera_handler = CameraHandler()
db_path = r'C:\ichun_test'


def load_model(model_path):
    model = YOLO(model_path)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model



product_01_model_path = r"C:\Projects\upload\MetalMarkAI\product_01\model_v1_gray.pt"


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

        # 取得矩陣大小
        product_matrix_row, product_matrix_column = map(int, data.get('matrix').split(','))
        # 取得產品料號

        product_code = data.get('product')
        date_info, time_info = data.get('op_time').split(' ')

        # absolute_path_to_db, image_file_name = generate_file_path_and_name(db_path, product_code, date_info, time_info)

        # print(f"Save image as name: {image_file_name}, in folder: {absolute_path_to_db}")

        # current_image = camera_handler.capture_image(absolute_path_to_db, image_file_name)

        # test_image = r"C:\Projects\upload\MetalMarkAI\product_01\test888.png"

        # if product_code in product_function_and_model_map:
        #     product_main_function, product_model = product_function_and_model_map[product_code]
        #     product_detected_result, product_execution_time = product_main_function(
        #         test_image,
        #         product_model,
        #         product_matrix_row,
        #         product_matrix_column
        #     )

        start_time = time.time()

        random_fail_row = random.randint(0, product_matrix_row - 1)
        random_fail_col = random.randint(0, product_matrix_column - 1)

        classification_results = {}
        for row in range(product_matrix_row):
            for column in range(product_matrix_column):
                if (row, column) == (random_fail_row, random_fail_col):
                    classification_results[(row, column)] = 'fail'
                else:
                    classification_results[(row, column)] = 'pass'

        time.sleep(random.uniform(0.5, 1.5))
        end_time = time.time()
        execution_time = end_time - start_time




        # 提取fail座標
        fail_coords = [f"{key[0]}, {key[1]}" for key, value in classification_results.items() if value == 'fail']

        # 判斷是否有fail的結果
        value_ary = fail_coords if fail_coords else "pass"

        return jsonify(
            {
                "Data": [
                    {
                        "stage": f"{detect_stage}",
                        "op_time": f"{date_info} {time_info}",
                        "product": f"{product_code}",
                        "details": f"{classification_results}"
                    }
                ],
                "ValueAry": value_ary,
                "timeTaken": f"{execution_time}"
            }
        )

    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(port=2486)

