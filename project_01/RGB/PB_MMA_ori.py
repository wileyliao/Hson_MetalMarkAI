import torch
import cv2
from torchvision import transforms
from ultralytics import YOLO

# GPU檢查
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 載入YOLO模型
model = YOLO('model_v1_gray.pt').to(device)
print("Model loaded to device")

# 圖片讀取 (在CPU上完成)
image_path = 'test888.png'
#image_path = ('./gray/cropped_0.png')
ori_img = cv2.imread(image_path)
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  # 轉成RGB
ori_img = cv2.resize(ori_img, (640, 640))


# 使用 transforms 將 OpenCV 的圖片轉成GPU張量
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(ori_img).unsqueeze(0).to(device)
print(f"Image tensor device: {img_tensor.device}")

# 進行檢測
results = model(img_tensor, conf = 0.4)
print("Detection completed on image")


# 處理檢測結果（在GPU上處理）
annotated_img = ori_img.copy()
metal_boxes = []
pass_fail_labels = []

for box in results[0].boxes:
    cls = int(box.cls)  # class index
    x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # xyxy coordinates (保持在GPU上處理)

    if cls == 1:  # metal框
        metal_boxes.append((x1, y1, x2, y2))
    elif cls == 0:  # fail框
        pass_fail_labels.append((x1, y1, x2, y2, 'fail'))
    elif cls == 2:  # pass框
        pass_fail_labels.append((x1, y1, x2, y2, 'pass'))

# 在metal框中間標記pass或fail (在CPU上完成)
for (mx1, my1, mx2, my2) in metal_boxes:
    label = "unknown"
    for (px1, py1, px2, py2, pf_label) in pass_fail_labels:
        if px1 >= mx1 and py1 >= my1 and px2 <= mx2 and py2 <= my2:
            label = pf_label
            break

    if label == "pass":
        color = (0, 255, 0)  # 綠色
    elif label == "fail":
        color = (255, 0, 0)  # 紅色
    else:
        color = (255, 0, 0)  # 默認紅色

    cv2.rectangle(annotated_img, (mx1, my1), (mx2, my2), color, 2)
    (text_width, text_height), _ = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 2)
    text_x = mx1 + (mx2 - mx1 - text_width) // 2
    text_y = my1 + (my2 - my1 + text_height) // 2
    cv2.putText(annotated_img, label.upper(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2, cv2.LINE_AA)

# 顯示結果 (在CPU上完成)
output_path = 'annotated_image.png'
cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
print(f"Annotated image saved to {output_path}")
