import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# 給定的線段數據
lines = np.array([[[27, 32, 36, 289]],
                  [[108, 27, 261, 22]],
                  [[39, 291, 250, 280]],
                  [[300, 43, 308, 276]],
                  [[36, 29, 260, 21]],
                  [[29, 126, 34, 284]],
                  [[302, 134, 305, 238]],
                  [[84, 291, 275, 281]],
                  [[29, 66, 37, 290]],
                  [[306, 225, 307, 280]]])

# 圖像的尺寸
width, height = 338, 326

# 存儲延伸後的線段
extended_lines = []

# 延伸線段到圖片邊界
for line in lines:
    x1, y1, x2, y2 = line[0]

    # 計算斜率和截距
    if x2 != x1:  # 避免垂直線（無限斜率）的情況
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1

        # 計算延伸到圖片邊界的交點
        y_at_x0 = m * 0 + c
        y_at_x_width = m * width + c
        x_at_y0 = (0 - c) / m
        x_at_y_height = (height - c) / m

        # 根據邊界條件選擇交點
        extended_points = []
        if 0 <= y_at_x0 <= height:
            extended_points.append((0, int(y_at_x0)))
        if 0 <= y_at_x_width <= height:
            extended_points.append((width, int(y_at_x_width)))
        if 0 <= x_at_y0 <= width:
            extended_points.append((int(x_at_y0), 0))
        if 0 <= x_at_y_height <= width:
            extended_points.append((int(x_at_y_height), height))

        # 選擇最靠近原線段的兩個點作為延伸線段的端點
        if len(extended_points) >= 2:
            extended_lines.append(extended_points[:2])

    else:  # 垂直線的情況
        extended_lines.append([(x1, 0), (x2, height)])


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
            if 0 <= point[0] <= width and 0 <= point[1] <= height:
                intersection_points.append(point)

# 去重並保留唯一的交點
unique_points = list(set(intersection_points))

# 找出最接近直角的矩形
def angle_between_points(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = b - a
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def is_right_angle(angle, tolerance=10):
    return 90 - tolerance <= angle <= 90 + tolerance

best_rectangle = None
best_score = float('inf')

for quad in combinations(unique_points, 4):
    angles = [
        angle_between_points(quad[0], quad[1], quad[2]),
        angle_between_points(quad[1], quad[2], quad[3]),
        angle_between_points(quad[2], quad[3], quad[0]),
        angle_between_points(quad[3], quad[0], quad[1])
    ]
    score = sum(abs(90 - angle) for angle in angles)
    if score < best_score:
        best_score = score
        best_rectangle = quad

if best_rectangle:
    rectangle_points = best_rectangle
else:
    rectangle_points = unique_points[:4]

# 計算矩形的邊長
def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

side_lengths = []
for i in range(len(rectangle_points)):
    p1 = rectangle_points[i]
    p2 = rectangle_points[(i + 1) % len(rectangle_points)]
    side_lengths.append(calculate_distance(p1, p2))

# 顯示結果
print("矩形頂點:\n", rectangle_points)
print("邊長:\n", side_lengths)

# 在圖像上繪製結果
image = np.ones((height, width, 3), dtype=np.uint8) * 255
image_extended_lines = image.copy()
image_rectangle_points = image.copy()

# 繪製延伸後的線段
for line in extended_lines:
    (x1, y1), (x2, y2) = line
    cv2.line(image_extended_lines, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

# 繪製矩形的頂點和邊
for point in rectangle_points:
    cv2.circle(image_rectangle_points, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)

for i in range(len(rectangle_points)):
    p1 = tuple(rectangle_points[i])
    p2 = tuple(rectangle_points[(i + 1) % len(rectangle_points)])
    # 繪製邊
    cv2.line(image_rectangle_points, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 1)
    # 計算邊的中點
    mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    # 在中點標註邊長
    cv2.putText(image_rectangle_points, f'{side_lengths[i]:.2f}', (int(mid_point[0]), int(mid_point[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
axs.imshow(image_extended_lines)
axs.set_title("Extended Lines")
axs.axis('off')
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(6, 6))
axs.imshow(image_rectangle_points)
axs.set_title("Rectangle Points")
axs.axis('off')
plt.show()
