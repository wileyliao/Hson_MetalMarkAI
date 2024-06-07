from ultralytics import YOLO
import torch
import cv2
def load_model(model_path):
    model = YOLO(model_path)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

def detect_objects(model, image_tensor, debug_mode = False):
    """
    使用YOLO模型檢測圖像中的物體。
    :param
        model:預訓練的YOLO檢測模型。
        image_tensor:需要檢測的圖像的PyTorch張量。
    :return:
        torch.Tensor: 檢測到的物件張量。
    """
    try:
        results = model(image_tensor)
        boxes = results[0].boxes.data
        metal_boxes = boxes[boxes[:, 5] == 1]
    except Exception as e:
        print(f"Error in detecting objects: {e}")
        raise

    if debug_mode:
        print(f"Detected metal boxes: {metal_boxes}")
        testannotated_img = results[0].plot()
        cv2.imshow("Detection Result", testannotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return metal_boxes

def classify_metal(model, metal, threshold, debug_mode=False):
    """
    將金屬圖像送入分類模型並提取分類結果。

    參數:
        model (torch.nn.Module): 預訓練的分類模型。
        metal (np.ndarray): 金屬圖像。
        threshold (float): 分類置信度的閾值。
        debug_mode (bool): 如果為True，顯示中間結果圖像。

    返回:
        tuple: 包含分類和置信度的元組。
    """
    try:
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

    except Exception as e:
        print(f"Error in classifying metal: {e}")
        raise

    return classification, confidence

def classify_and_mark_images(model, metals_with_positions, rows, cols, threshold, classify_fn, debug_mode=False):
    """
    使用分類模型對金屬進行分類並標記。

    參數:
        model (torch.nn.Module): 預訓練的分類模型。
        metals_with_positions (list): 含有金屬圖像和位置的列表。
        rows (int): 圖像的行數。
        cols (int): 圖像的列數。
        threshold (float): 分類置信度的閾值。
        classify_fn (function): 用於分類金屬的函數。
        debug_mode (bool): 如果為True，顯示中間結果圖像。

    返回:
        dict: 包含每個位置的分類結果。
    """
    classification_results = {(r, c): "empty" for r in range(rows) for c in range(cols)}

    for i, (metal, (row, col)) in enumerate(metals_with_positions):
        try:
            classification, confidence = classify_fn(model, metal, threshold, debug_mode)
            if classification in ["fail", "pass"]:
                classification_results[(row, col)] = {
                    "classification": classification,
                    "confidence": confidence
                }
            else:
                classification_results[(row, col)] = "empty"

        except Exception as e:
            print(f"Error processing metal at position ({row}, {col}): {e}")

    return classification_results
