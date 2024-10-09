import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T


def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f'{func.__name__}: {e}'
            raise RuntimeError(error_message)
    return wrapper


@error_handler
def preprocess_image(image_path, resize_width, resize_height, debug_mode=False):
    """
    讀取圖像並進行大小調整，將其轉換為PyTorch張量並移到GPU。
    :param
        image_path: 原始影像路徑
        resize_width: CNN Model接受的輸入大小
        resize_height: CNN Model接受的輸入大小
    :return
        ori_tensor (torch.Tensor): 原始圖像的PyTorch張量表示。
        rez_ori_tensor (torch.Tensor): 調整大小後的圖像的PyTorch張量表示。
    """

    ori = cv2.imread(image_path)
    # 將圖像轉換為 PyTorch 張量並移到 GPU
    ori_tensor = torch.from_numpy(ori).permute(2, 0, 1).float().cuda()
    # 在 GPU 上進行重新調整大小
    resize_transform = T.Resize((resize_height, resize_width))
    rez_ori_tensor = resize_transform(ori_tensor)

    if debug_mode:
        print(f"Original tensor shape: {ori_tensor.shape}")
        print(f"Resized tensor shape: {rez_ori_tensor.shape}")
        # 將原圖轉回 CPU 並顯示
        ori = ori_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # 將重新調整大小的圖像轉回 CPU 並顯示
        rez_ori = rez_ori_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        cv2.imshow("Original Image", ori)
        cv2.imshow("Resized Image", rez_ori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return ori_tensor, rez_ori_tensor


@error_handler
def map_boxes(boxes, ori_tensor, rez_tensor, rows, cols, output_dir, debug_mode=False):
    """
    將檢測到的boxes映射回原始影像，並計算其所對應的矩陣位置
    :param
        boxes:
        ori_tensor(torch.Tensor): 原始圖像的張量
        rez_tensor(torch.Tensor): 調整大小後的圖像的張量
        rows: 矩陣的行數
        cols: 矩陣的列數
        output_dir: 用於保存結果的目錄
    :return
        torch.Tensor: 包含映射位置的框
    """

    ori_height, ori_width = ori_tensor.shape[1:3]
    rez_height, rez_width = rez_tensor.shape[1:3]
    scale_x = ori_width / rez_width
    scale_y = ori_height / rez_height
    mapped_boxes = boxes.clone()
    mapped_boxes[:, [0, 2]] *= scale_x
    mapped_boxes[:, [1, 3]] *= scale_y

    '''Matrix'''
    # 計算中心座標
    centers = (mapped_boxes[:, :2] + mapped_boxes[:, 2:4]) / 2
    # 確定每個區域的寬度和高度
    cell_width = ori_width / cols
    cell_height = ori_height / rows
    # 計算每個中心點對應的矩陣位置
    col_indices = (centers[:, 0] // cell_width).long()  # 使用整數下取
    row_indices = (centers[:, 1] // cell_height).long()

    # 將矩陣位置添加到 mapped_boxes 張量中
    positions = torch.stack((row_indices, col_indices), dim=1).float().to(boxes.device)  # 保持在 GPU 上
    mapped_boxes_with_positions = torch.cat((mapped_boxes, positions), dim=1)
    '''Matrix'''

    if debug_mode:
        ori_cpu = ori_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rez_cpu = rez_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        ori_cpu = np.ascontiguousarray(ori_cpu, dtype=np.uint8)
        rez_cpu = np.ascontiguousarray(rez_cpu, dtype=np.uint8)

        for box in mapped_boxes.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(ori_cpu, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for box in boxes.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(rez_cpu, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 視覺化
        cv2.imwrite(os.path.join(output_dir, "01_Original_results_gpu.png"), ori_cpu)
        cv2.imwrite(os.path.join(output_dir, "02_Resized_Results_gpu.png"), rez_cpu)
        cv2.imshow("01_Original_results_gpu", ori_cpu)
        cv2.imshow("02_Resized_Results_gpu", rez_cpu)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mapped_boxes_with_positions


@error_handler
def cut_metal_from_boxes(mapped_boxes_between_images, ori_tensor, resize_width, resize_height, debug_mode=False):
    """
    將物件從圖像中裁剪出來並調整大小
    :param
        mapped_boxes_between_images (torch.Tensor): 映射到原圖像的框位置。
        ori_tensor (torch.Tensor): 原始圖像的張量表示。
        resize_width (int): 調整後的圖像寬度。
        resize_height (int): 調整後的圖像高度。
        debug_mode (bool): 如果為True，顯示中間結果圖像。
    :return:
        list: 含有物件和對應矩陣位置的列表。
    """
    metals_with_positions = []
    resize_transform = T.Resize((resize_height, resize_width))

    for i, box in enumerate(mapped_boxes_between_images):
        x1, y1, x2, y2, conf, cls, row, col = map(int, box[:8])  # 取8個元素，其中row和col是矩陣位置
        cropped_metal = ori_tensor[:, y1:y2, x1:x2]  # 保持在GPU上
        resize_ori_metal = resize_transform(cropped_metal)

        rez_ori_metal_cpu = resize_ori_metal.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rez_ori_metal_cpu = np.ascontiguousarray(rez_ori_metal_cpu, dtype=np.uint8)

        # 將調整後的 metal 和它的矩陣位置一起儲存
        metals_with_positions.append((rez_ori_metal_cpu, (row, col)))

        if debug_mode:
            print(f"Cut metal {i} at position ({row}, {col})")

    return metals_with_positions

# def show_result(rows, columns, metals_with_positions):

