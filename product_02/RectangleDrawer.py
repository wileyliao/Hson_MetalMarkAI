import cv2

class RectangleDrawer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.rect = []
        self.drawing = False

        if self.img is None:
            raise ValueError(f"Failed to load image from {image_path}")

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 滑鼠按下左鍵
            self.rect = [(x, y)]  # 紀錄起始位置
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:  # 滑鼠移動
            if self.drawing:
                img_copy = self.img.copy()
                cv2.rectangle(img_copy, self.rect[0], (x, y), (0, 255, 0), 2)
                cv2.imshow("image", img_copy)
        elif event == cv2.EVENT_LBUTTONUP:  # 滑鼠鬆開左鍵
            self.rect.append((x, y))  # 紀錄終止點
            self.drawing = False
            cv2.rectangle(self.img, self.rect[0], self.rect[1], (0, 255, 0), 2)
            cv2.imshow("image", self.img)
            print(f"Selected Rectangle Coordinates: {self.rect}")

    def start_drawing(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.draw_rectangle)

        while True:
            cv2.imshow("image", self.img)
            if cv2.waitKey(1) & 0xFF == 27:  # 按下 ESC 退出
                break
        cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":

    drawer = RectangleDrawer('C:/Projects/MetalMarkAI/captured_image_0.png')

    drawer.start_drawing()
