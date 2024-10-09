from ultralytics import YOLO
import multiprocessing
import torch


def env_check():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

def class_check():
    model = YOLO('model_v1.pt')
    for class_id, class_name in model.names.items():
        print(f"ID: {class_id}, Class: {class_name}")

def model_train():
    # model storage path = my_project/test_run_01
    multiprocessing.freeze_support()
    model = YOLO('models/yolov8n.pt')
    results = model.train(
        data = 'data.yaml',
        imgsz = 256,
        epochs = 200,
        patience = 10,
        batch = 10,
        project = 'my_project',
        name = 'test_run_01'
    )
    return results

if __name__ == "__main__":
    env_check()