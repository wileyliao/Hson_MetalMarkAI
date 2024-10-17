from .product_03_main_utils import *
from .product_03_main_return_data import calculate_matrix_positions_and_relations
from ultralytics import YOLO
import time
import cv2

"""note
ID: 0, Class: door
ID: 1, Class: point
"""


def product_03_main(image_from_camera, model_global, model_local):

    start_time = time.time()
    image_global = cv2.imread(image_from_camera)

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

    end_time = time.time()

    execution_time = end_time - start_time

    return matrix_relations, execution_time


if __name__ == '__main__':
    image_path = r"./data/global/origin/008.png"

    model_g = YOLO(r"./models/global.pt")
    model_l = YOLO(r"./models/local.pt")

    result, exe_time = product_03_main(image_path, model_g, model_l)
    print(f'result: \n {result}')
    print(f'time taken: {exe_time}')
