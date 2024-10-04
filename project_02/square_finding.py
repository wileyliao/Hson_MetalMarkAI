import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# 假設已經有延伸後的線段和交點
extended_lines = [[(0, 151), (331, 145)],
                  [(0, 18), (331, 12)],
                  [(0, 16), (331, 10)],
                  [(0, 149), (331, 144)],
                  [(7, 0), (10, 157)],
                  [(294, 0), (297, 157)],
                  [(331, 149), (218, 157)]]


# 計算兩條線段的交點
def line_intersection(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 平行或重疊，無交點

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return (px, py)


# 找出所有線段之間的交點
intersection_points = []
for i in range(len(extended_lines)):
    for j in range(i + 1, len(extended_lines)):
        point = line_intersection(extended_lines[i], extended_lines[j])
        if point is not None:
            # 檢查交點是否在圖片範圍內
            if 0 <= point[0] <= 331 and 0 <= point[1] <= 157:
                intersection_points.append(point)

# 去重並保留唯一的交點
unique_points = list(set(intersection_points))

# 使用凸包找到最小多邊形
if len(unique_points) >= 4:
    points = np.array(unique_points)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
else:
    hull_points = unique_points


# 計算邊長
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


side_lengths = []
for i in range(len(hull_points)):
    p1 = hull_points[i]
    p2 = hull_points[(i + 1) % len(hull_points)]
    side_lengths.append(calculate_distance(p1, p2))

# 顯示結果
print("交點:", hull_points)
print("邊長:", side_lengths)

# 在圖像上繪製結果
image = np.ones((157, 331, 3), dtype=np.uint8) * 255

# 繪製延伸後的線段
for line in extended_lines:
    (x1, y1), (x2, y2) = line
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

# 繪製交點和凸包
for point in hull_points:
    cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

for i in range(len(hull_points)):
    p1 = tuple(hull_points[i])
    p2 = tuple(hull_points[(i + 1) % len(hull_points)])
    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 2)

# 顯示結果
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.title("延伸後的線段、交點和多邊形")
plt.axis('off')
plt.show()
