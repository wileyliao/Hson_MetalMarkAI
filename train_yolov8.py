import ultralytics
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO('models/yolov8n.pt')
    results = model.train(
        data = 'data.yaml',
        imgsz = 256,
        epochs = 200,
        patience = 10,
        batch = 10,
        project = 'yolov8_object_1',
        name = 'exp01'
    )