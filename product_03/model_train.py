import ultralytics
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO('models/yolov8n.pt')
    result = model.train(
        data="./models/label/local/data.yaml",
        imgsz=640,
        epochs=100,
        patience=20,
        batch=16,
        project='product_03_local',
        name='exp_01'
    )
