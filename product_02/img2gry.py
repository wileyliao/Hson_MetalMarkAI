import cv2
import matplotlib.pyplot as plt

# 讀取彩色圖像
image = cv2.imread('test.jpg')  # 替換為你的圖像路徑

# 將圖像從 BGR 轉換為灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顯示原圖和灰度圖像
plt.figure(figsize=(10, 5))

# 灰度圖顯示
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray Image')
plt.axis('off')

plt.show()

# 保存灰度圖像到文件
cv2.imwrite('gray_image.jpg', gray_image)