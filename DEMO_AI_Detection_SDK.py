import cv2

def process_frame(frame, model, confidence_thresholld):
    results = model(frame)
    boxes_by_class = {'metal': [], 'pass': [], 'fail': []}
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = map(int, box)
            class_name = result.names[class_id]
            if class_name in boxes_by_class:
                boxes_by_class[class_name].append((x1, y1, x2, y2))
                color = (0, 0, 255) if class_name == 'fail' else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return boxes_by_class

