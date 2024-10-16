from ultralytics import YOLO
import cv2
import os

model_global = YOLO(r"C:\Projects\upload\MetalMarkAI\product_03\models\global.pt")
model_local = YOLO(r"C:\Projects\upload\MetalMarkAI\product_03\models\local.pt")


def image_padding(image, pad_param):
    image_height, image_width = image.shape[:2]
    pad_top = (pad_param - image_height) // 2
    pad_bottom = pad_param - image_height - pad_top
    pad_left = (pad_param - image_width) // 2
    pad_right = pad_param - image_width - pad_left

    image_padded = cv2.copyMakeBorder(
        image_origin, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return image_padded



if __name__ == '__main__':
    image_path = r"C:/Projects/upload/MetalMarkAI/product_03/data/local/origin/078.png"

    image_origin = cv2.imread(image_path)

    image_pad = image_padding(image_origin, 1280)

    image_resized = cv2.resize(image_pad, (640,640))

    results = model_local(image_resized, conf=0.5)
    results_show = results[0].plot()

    cv2.imshow('result', results_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # os.makedirs(output_folder, exist_ok=True)
    #
    # for idx, box in enumerate(results[0].boxes.xyxy):
    #     # 獲取偵測框的座標
    #     x1_resized, y1_resized, x2_resized, y2_resized = box
    #
    #     # 計算縮放比例，從 640x640 映射回 3200x3200
    #     scale_factor_x = 3200 / 640
    #     scale_factor_y = 3200 / 640
    #
    #     # 映射回 `image_padded` 中的座標
    #     x1_padded = int(x1_resized * scale_factor_x)
    #     y1_padded = int(y1_resized * scale_factor_y)
    #     x2_padded = int(x2_resized * scale_factor_x)
    #     y2_padded = int(y2_resized * scale_factor_y)
    #
    #     # 裁剪出物體區域
    #     cropped_image = image_padded[y1_padded:y2_padded, x1_padded:x2_padded]
    #
    #     # 構建保存的路徑，按索引命名
    #     crop_output_path = os.path.join(output_folder, f"cropped_{idx}.png")
    #
    #     # 保存裁剪的影像
    #     cv2.imwrite(crop_output_path, cropped_image)
    #     print(f"已保存裁剪影像: {crop_output_path}")
