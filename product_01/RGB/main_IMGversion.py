import cv2
from ultralytics import YOLO



model = YOLO('model_v1.pt')
img_path = 'test004.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (640, 640))
results = model(img)

boxes_by_class = {'metal': [], 'pass': [], 'fail': []}
class_ids = {'metal': 1, 'pass': 2, 'fail': 0}

for result in results:
    for box in result.boxes.data:
        x1, y1, x2, y2, score, class_id = map(int, box)
        class_name = result.names[class_id]
        if class_name in boxes_by_class:
            boxes_by_class[class_name].append((x1, y1, x2, y2))
            color = (0, 0, 255) if class_name == 'fail' else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def check_inside(box1, box2):
    x1, y1, x2, y2 = box1
    x1i, y1i, x2i, y2i = box2
    return x1 <= x1i and x2 >= x2i and y1 <= y1i and y2 >= y2i

inside_info = [
    (metal_box, label)
    for metal_box in boxes_by_class['metal']
    for label in ['pass', 'fail']
    for box in boxes_by_class[label]
    if check_inside(metal_box, box)
]

for box, label in inside_info:
    mx1, my1, mx2, my2 = box
    center_x = (mx1 + mx2) // 2
    center_y = (my1 + my2) // 2
    text = label.upper()
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    start_x = center_x - text_size[0] // 2
    start_y = center_y + text_size[1] // 2
    text_color = (0, 0, 255) if label == 'fail' else (0, 255, 0)
    cv2.putText(img, text, (start_x, start_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)

# 显示图像
cv2.imshow('YOLOv8 Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()