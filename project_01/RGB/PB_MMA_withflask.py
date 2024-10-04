import torch
import cv2
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt


def load_model(model_path, device):
    model = YOLO(model_path).to(device)
    print("Model loaded to device")
    return model


def preprocess_image(image_path, resize_size):
    ori_img = cv2.imread(image_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(ori_img, resize_size)
    transform = transforms.Compose([transforms.ToTensor()])
    resized_img_tensor = transform(resized_img).unsqueeze(0).to(device)
    return ori_img, resized_img_tensor


def detect_objects(model, image_tensor):
    results = model(image_tensor)
    return results


def get_class_boxes(results, class_id):
    boxes = []
    for box in results[0].boxes:
        if box.cls == class_id:
            boxes.append(box)
    return boxes


def scale_boxes(boxes, scale_x, scale_y):
    scaled_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        scaled_boxes.append([int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)])
    return scaled_boxes


def crop_and_resize_image(image, box, resize_size):
    cropped_img = image[box[1]:box[3], box[0]:box[2]]
    cropped_resized_img = cv2.resize(cropped_img, resize_size)
    transform = transforms.Compose([transforms.ToTensor()])
    cropped_resized_img_tensor = transform(cropped_resized_img).unsqueeze(0).to(device)
    return cropped_resized_img_tensor


def plot_results(image, boxes, labels):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Original Image with class1 boxes")
    for idx, box in enumerate(boxes):
        plt.gca().add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor='red', facecolor='none',
                          linewidth=2))
        plt.text((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, labels[idx], color='white', fontsize=12, ha='center',
                 va='center', bbox=dict(facecolor='red', alpha=0.5))
    plt.show()


# 主程式
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = 'model_v1.pt'
image_path = 'test999.jpg'
resize_size = (640, 640)

model = load_model(model_path, device)
ori_img, resized_img_tensor = preprocess_image(image_path, resize_size)
results = detect_objects(model, resized_img_tensor)
class1_boxes = get_class_boxes(results, 1)

print(f"Found {len(class1_boxes)} class1 boxes")

ori_height, ori_width = ori_img.shape[:2]
resize_height, resize_width = resized_img_tensor.shape[2], resized_img_tensor.shape[3]
scale_x = ori_width / resize_width
scale_y = ori_height / resize_height

scaled_class1_boxes = scale_boxes(class1_boxes, scale_x, scale_y)
class1_labels = []

for idx, box in enumerate(scaled_class1_boxes):
    cropped_resized_img_tensor = crop_and_resize_image(ori_img, box, resize_size)

    # 轉換為可視化圖像
    cropped_resized_img = cropped_resized_img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.figure()
    plt.imshow(cropped_resized_img)
    plt.title(f"Cropped Image {idx}")
    plt.show()

    cropped_results = detect_objects(model, cropped_resized_img_tensor)

    label = "unknown"
    for detection in cropped_results[0].boxes:
        if detection.cls == 0:
            label = "fail"
        elif detection.cls == 2:
            label = "pass"
    class1_labels.append(label)
    print(f"Box {idx}: label={label}")

plot_results(ori_img, scaled_class1_boxes, class1_labels)
