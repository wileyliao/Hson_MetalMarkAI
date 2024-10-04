import torch
import cv2
from torchvision import transforms
from ultralytics import YOLO
from flask import Flask, request, jsonify
from logging.handlers import RotatingFileHandler
import os
import logging

app = Flask(__name__)

def setup_logger(model_name):
    # 確保 logs 目錄存在
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 根據 model 名稱設置 LOG 文件名稱
    log_file = os.path.join(log_directory, f'{model_name}.log')

    # 設置 RotatingFileHandler，最大文件大小為 5MB，保留 3 個備份
    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
    handler.setLevel(logging.INFO)

    # 設置 LOG 格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # 設置 logger
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


def check_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def load_model(model_name, device):
    if not model_name.endswith('.pt'):
        model_name += '.pt'
    model = YOLO(model_name).to(device)
    print("Model loaded to device")
    return model

def capture_image(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Unable to open camera with ID {camera_id}")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to capture image from camera")

    cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = cv2.resize(frame, (640, 640))
    return frame

def preprocess_image(image, device):
    # 灰階影像已轉換為 3 通道 RGB 影像
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0).to(device)
    print(f"Image tensor device: {img_tensor.device}")
    return img_tensor


def map_yolo_to_original(yolo_box, original_shape, yolo_shape=(640, 640)):
    """
    將 YOLO 檢測框的座標映射回原始圖像的座標。
    """
    x_ratio = original_shape[1] / yolo_shape[1]
    y_ratio = original_shape[0] / yolo_shape[0]

    x1, y1, x2, y2 = yolo_box
    x1 = int(x1 * x_ratio)
    y1 = int(y1 * y_ratio)
    x2 = int(x2 * x_ratio)
    y2 = int(y2 * y_ratio)

    return x1, y1, x2, y2

# def load_and_preprocess_image(image_path, device):
#     ori_img = cv2.imread(image_path)
#     ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
#     ori_img = cv2.resize(ori_img, (640, 640))
#     transform = transforms.Compose([transforms.ToTensor()])
#     img_tensor = transform(ori_img).unsqueeze(0).to(device)
#     print(f"Image tensor device: {img_tensor.device}")
#     return ori_img, img_tensor

def detect_image(model, img_tensor, ori_img, device, output_dir="output"):
    results = model(img_tensor)
    print("Detection completed on image")

    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 找到 class 1 的金屬框，並在原圖中裁剪相應區域進行二次檢測
    metal_boxes = [box.xyxy[0].int().tolist() for box in results[0].boxes if int(box.cls) == 1]

    pass_fail_labels = []

    for i, metal_box in enumerate(metal_boxes):
        # 映射座標到原圖
        x1, y1, x2, y2 = map_yolo_to_original(metal_box, ori_img.shape)
        cropped_img = ori_img[y1:y2, x1:x2]

        # 將裁剪出的圖像轉換為 3 通道 RGB 影像並調整大小為 640x640
        cropped_rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2RGB)
        resized_cropped_img = cv2.resize(cropped_rgb_img, (640, 640))

        # 將調整大小後的圖像保存
        cropped_img_path = os.path.join(output_dir, f"cropped_{i}.png")
        cv2.imwrite(cropped_img_path, cv2.cvtColor(resized_cropped_img, cv2.COLOR_RGB2BGR))
        print(f"Cropped image saved to {cropped_img_path}")

        # 將調整大小後的圖像轉換為張量
        cropped_tensor = transforms.ToTensor()(resized_cropped_img).unsqueeze(0).to(device)

        # 二次 YOLO 檢測
        secondary_results = model(cropped_tensor)
        for box in secondary_results[0].boxes:
            cls = int(box.cls)
            if cls == 0 or cls == 2:
                # 映射座標回原圖
                sx1, sy1, sx2, sy2 = box.xyxy[0].int().tolist()
                original_sx1 = int(sx1 * (x2 - x1) / 640)
                original_sy1 = int(sy1 * (y2 - y1) / 640)
                original_sx2 = int(sx2 * (x2 - x1) / 640)
                original_sy2 = int(sy2 * (y2 - y1) / 640)
                pass_fail_labels.append((x1 + original_sx1, y1 + original_sy1, x1 + original_sx2, y1 + original_sy2, 'pass' if cls == 2 else 'fail'))
                break

    return metal_boxes, pass_fail_labels



def process_detection_results(results):
    metal_boxes = []
    pass_fail_labels = []

    for box in results[0].boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()

        if cls == 1:
            metal_boxes.append((x1, y1, x2, y2))
        elif cls == 0:
            pass_fail_labels.append((x1, y1, x2, y2, 'fail'))
        elif cls == 2:
            pass_fail_labels.append((x1, y1, x2, y2, 'pass'))

    return metal_boxes, pass_fail_labels


def annotate_image(ori_img, metal_boxes, pass_fail_labels):
    annotated_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2RGB)  # 確保輸出的圖像為 3 通道 RGB 圖像
    for (mx1, my1, mx2, my2) in metal_boxes:
        label = "unknown"
        for (px1, py1, px2, py2, pf_label) in pass_fail_labels:
            # 檢查是否標記框在金屬框內
            if px1 >= mx1 and py1 >= my1 and px2 <= mx2 and py2 <= my2:
                label = pf_label
                break

        color = (0, 255, 0) if label == "pass" else (255, 0, 0)
        cv2.rectangle(annotated_img, (mx1, my1), (mx2, my2), color, 2)

        # 計算文字的位置，確保文字在框內且居中
        (text_width, text_height), _ = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 2)
        text_x = mx1 + (mx2 - mx1 - text_width) // 2
        text_y = my1 + (my2 - my1 + text_height) // 2

        cv2.putText(annotated_img, label.upper(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2, cv2.LINE_AA)

    return annotated_img



def save_annotated_image(annotated_img, output_path):
    cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    print(f"Annotated image saved to {output_path}")


def calculate_center(box):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy


def determine_matrix_positions(metal_boxes, pass_fail_labels, rows, cols):
    # 計算每個框的中心座標
    image_width = 640
    image_height = 640
    centers = [(box, calculate_center(box)) for box in metal_boxes]

    # 確定每個區域的寬度和高度
    cell_width = image_width / cols
    cell_height = image_height / rows

    # 將框分配到相應的矩陣位置
    matrix_positions = {}
    matrix_labels = {}
    all_positions = [(r, c) for c in range(cols) for r in range(rows)]
    used_positions = set()

    for box, (cx, cy) in centers:
        col = int(cx // cell_width)
        row = int(cy // cell_height)
        matrix_positions[(row, col)] = box
        used_positions.add((row, col))

        # 確定標籤
        label = "unknown"
        for (px1, py1, px2, py2, pf_label) in pass_fail_labels:
            if px1 >= box[0] and py1 >= box[1] and px2 <= box[2] and py2 <= box[3]:
                label = pf_label
                break
        matrix_labels[(row, col)] = label

        # 找到空的格子並設置標籤為unknown
        empty_positions = set(all_positions) - used_positions
        for pos in empty_positions:
            matrix_labels[pos] = "unknown"

    return matrix_positions, empty_positions, matrix_labels


@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    if data['detection'] != 'start':
        return jsonify({"error": "Invalid detection command"}), 400
    if 'cameraID' not in data or 'model' not in data:
        return jsonify({"error": "Please provide both cameraID and model"}), 400

    camera_id = int(data['cameraID'])
    model_name = data['model']
    logger = setup_logger(model_name)

    try:
        device = check_device()
        logger.info("--------------------------------------------")
        logger.info("--------Detection command received----------")
        logger.info("--------------------------------------------")
        logger.info(f"-------Using device: {device}")

        model = load_model(model_name, device)
        logger.info(f"--GPU--Model {model_name} loaded to device")

        image_path = 'test888.png'
        ori_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 讀取灰階圖像
        rez_img = cv2.resize(ori_img, (640, 640))

        # 將灰階影像轉換為 3 通道的 RGB 影像以適應模型輸入
        rgb_img = cv2.cvtColor(rez_img, cv2.COLOR_GRAY2RGB)

        img_tensor = preprocess_image(rgb_img, device)
        logger.info(f"-------Image preprocessed and moved to {device}")

        # 檢測圖像
        metal_boxes, pass_fail_labels = detect_image(model, img_tensor, ori_img, device)
        logger.info("-------------------------------")
        logger.info("---Detection completed on image")
        logger.info(f"---Detection results processed: {len(metal_boxes)} metal boxes detected")

        # 確定矩陣位置和標籤
        rows, cols = 3, 4
        matrix_positions, empty_positions, matrix_labels = determine_matrix_positions(metal_boxes, pass_fail_labels, rows, cols)
        logger.info("---Matrix positions and labels determined")

        # 標註圖像
        annotated_img = annotate_image(ori_img, metal_boxes, pass_fail_labels)
        save_annotated_image(annotated_img, 'annotated_image.png')
        logger.info("---Annotated image saved to 'annotated_image.png'")

        matrix_result = {
            "matrix_labels": {str(pos): label for pos, label in matrix_labels.items()}
        }

        for pos, box in matrix_positions.items():
            logger.info(f"Position {pos}: {box} - Label: {matrix_labels[pos]}")
        for pos in empty_positions:
            logger.info(f"Empty position: {pos} - Label: unknown")

        return jsonify(
            {
                "result": matrix_result
            }
        )

    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
