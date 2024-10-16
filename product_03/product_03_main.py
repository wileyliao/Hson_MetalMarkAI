from product_03_main_utils import *
from product_03_main_return_data import calculate_matrix_positions_and_relations
from ultralytics import YOLO
import cv2
import json

with open('path_config.json', 'r') as file:
    path_config = json.load(file)


model_global = YOLO(path_config["model_global_path"])
model_local = YOLO(path_config["model_local_path"])
image_path = r"./data/global/origin/008.png"


"""note
ID: 0, Class: door
ID: 1, Class: point
"""


def main():

    image_global = cv2.imread(image_path)

    image_global_padding = image_padding(image_global, 3200)
    image_global_padding_resized = cv2.resize(image_global_padding, (640, 640))

    image_global_padding_resized_result = model_global(image_global_padding_resized, conf=0.5)

    # image_global_padding_resized_result_show = image_global_padding_resized_result[0].plot()
    # cv2.imshow('image_global_padding_resized_result_show', image_global_padding_resized_result_show)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    images_local, images_local_boxes = map_and_crop(image_global_padding_resized_result, image_global_padding)

    images_local_results_dict = process_cropped_images(images_local, model_local)

    matrix_relations = calculate_matrix_positions_and_relations(images_local_boxes, images_local_results_dict)

    return matrix_relations


if __name__ == '__main__':
    print(main())
