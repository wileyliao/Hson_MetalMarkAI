from flask import Flask, request, jsonify
from ultralytics import YOLO
from ultralytics.nn.tasks import temporary_modules

from Camera_Control import CameraHandler
from File_Management import generate_file_path_and_name
from product_01.product_01_main import product_01_main
from product_03.product_03_main import product_03_main
import json
import os

app = Flask(__name__)


with open(r"C:\Projects\upload\MetalMarkAI\path_config.json", 'r') as file:
    path_config = json.load(file)


product_function_model_map = {
    'product_03': (
        product_03_main,
        YOLO(path_config["product_03_model_global_path"]),
        YOLO(path_config["product_03_model_local_path"])
    )
}

db_path = os.path.normpath(path_config["product_db_path"])


product_03_temporary_folder = os.path.normpath(path_config["product_03_temporary_folder"])
product_03_test_usage_image_path = os.path.normpath(path_config["product_03_test_image_path"])



@app.route('/MetalMarkAI', methods=['POST'])
def main():
    try:
        # camera_handler = CameraHandler()
        # Receive data
        data = request.json['Data'][0]
        detect_stage = data.get('stage')
        product_code = data.get('product')
        date_info, time_info = data.get('op_time').split(' ')

        # Get image form camera & write into disk
        abs_path_to_db, image_file_name = generate_file_path_and_name(db_path, product_code, date_info, time_info)
        # current_image = camera_handler.capture_image(product_03_temporary_folder, detect_stage, abs_path_to_db, image_file_name)

        product_detected_result = {}
        product_execution_time = 0

        # Algorithm decision
        if product_code in product_function_model_map:
            product_main_function, model_global, model_local = product_function_model_map[product_code]

            product_detected_result, product_execution_time = product_main_function(
                detect_stage,
                product_03_test_usage_image_path,
                product_03_temporary_folder,
                model_global,
                model_local,
                abs_path_to_db,
                image_file_name
            )

        # Extract Fail data
        fail_coords = [f"{key[0]}, {key[1]}" for key, value in product_detected_result.items() if value == 'fail']
        value_ary = fail_coords if fail_coords else None

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
    app.run(port=2486, debug=True)

