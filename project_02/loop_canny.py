import cv2
from matplotlib import pyplot as plt

# 讀取圖像
image = cv2.imread('./box_coord/metal01.jpg', cv2.IMREAD_GRAYSCALE)

# 定義範圍
low_threshold_range = range(0, 40, 10)
high_threshold_range = range(40, 100, 10)

# 建立圖像顯示窗口
plt.figure(figsize=(20, 20))

# 使用雙重迴圈遍歷不同的閾值組合
plot_index = 1
for low in low_threshold_range:
    for high in high_threshold_range:
        edges = cv2.Canny(image, low, high)

        plt.subplot(len(low_threshold_range), len(high_threshold_range), plot_index)
        plt.imshow(edges, cmap='gray')
        plt.title(f'Low: {low}, High: {high}')
        plt.xticks([]), plt.yticks([])
        plot_index += 1

plt.show()
