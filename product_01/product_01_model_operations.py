from ultralytics import YOLO
import torch
import cv2


def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f'{func.__name__}: {e}'
            raise RuntimeError(error_message)
    return wrapper


@error_handler
def load_model(model_path):
    model = YOLO(model_path)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model


@error_handler
def detect_objects(model, image_tensor, debug_mode = False):
    """
    使用YOLO模型檢測圖像中的物體。
    :param
        model:預訓練的YOLO檢測模型。
        image_tensor:需要檢測的圖像的PyTorch張量。
    :return:
        torch.Tensor: 檢測到的物件張量。
    """

    results = model(image_tensor)
    boxes = results[0].boxes.data
    metal_boxes = boxes[boxes[:, 5] == 1]

    if debug_mode:
        print(f"Detected metal boxes: {metal_boxes}")
        detected_img = results[0].plot()
        cv2.imshow("Detection Result", detected_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return metal_boxes


@error_handler
def classify_metal(model, metal, threshold, debug_mode=False):
    """
    將金屬圖像送入分類模型並提取分類結果。

    :param
        model (torch.nn.Module): 預訓練的分類模型。
        metal (np.ndarray): 金屬圖像。
        threshold (float): 分類置信度的閾值。
        debug_mode (bool): 如果為True，顯示中間結果圖像。

    :return
        tuple: 包含分類和置信度的元組。
    """

    results = model(metal, conf=threshold)
    classification = "unknown"
    confidence = 0.0

    if results[0].boxes:  # 檢查是否有檢測到的 boxes
        for box in results[0].boxes:
            cls = box.cls.item()  # 將 cls tensor 轉換為 Python 的數字類型
            conf = box.conf.item()  # 同樣處理 confidence
            class_name = results[0].names[cls]

            if debug_mode:
                print(f"Detected class index: {cls}, confidence: {conf}")
                print(f"Detected class name: {class_name}")

            if conf > threshold and class_name in ["fail", "pass"]:
                classification = class_name
                confidence = conf

    if debug_mode:
        annotated_img = results[0].plot()
        cv2.imshow(f"Detection Result - Classification: {classification}", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return classification, confidence


@error_handler
def classify_and_mark_images(model, metals_with_positions, threshold, debug_mode=False):
    """
    使用分類模型對金屬進行分類並標記。

    :param
        model (torch.nn.Module): 預訓練的分類模型。
        metals_with_positions (list): 含有金屬圖像和位置的列表。
        threshold (float): 分類置信度的閾值。
        classify_fn (function): 用於分類金屬的函數。
        debug_mode (bool): 如果為True，顯示中間結果圖像。

    :return
        dict: 包含每個位置的分類結果。
    """
    classification_results = {}

    for i, (metal, (row, col)) in enumerate(metals_with_positions):
        classification, confidence = classify_metal(model, metal, threshold, debug_mode)
        if classification in ["fail", "pass"]:
            classification_results[(row, col)] = classification
        else:
            classification_results[(row, col)] = "empty"

    return classification_results

