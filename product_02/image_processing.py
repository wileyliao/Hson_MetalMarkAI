import cv2
from shapely.geometry import LineString, Point
import math
import numpy as np
import pylab as plt
from sklearn.cluster import KMeans

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def compute_angle(line):
    x1, y1, x2, y2 = line
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle
def group_by_angle(lines, tolerance):
    groups = []
    for line in lines:
        angle = compute_angle(line)
        added = False
        for group in groups:
            if abs(group['angle'] - angle) <= tolerance:
                group['lines'].append(line)
                added = True
                break
        if not added:
            groups.append({'angle': angle, 'lines': [line]})
    return groups
def kmeans_cluster(lines, n_clusters):
    coords = np.array([[(line[0] + line[2]) / 2, (line[1] + line[3]) / 2] for line in lines])
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(coords)
    labels = kmeans.labels_
    clusters = [[] for _ in range(n_clusters)]
    for label, line in zip(labels, lines):
        clusters[label].append(line)
    return clusters
def compute_slope(line):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:  # 避免除以零
        return np.inf
    return (y2 - y1) / (x2 - x1)
def compute_new_line(slope, midpoint, length=1000):
    if abs(slope) == np.inf:
        x1, x2 = midpoint, midpoint
        y1, y2 = midpoint - length / 2, midpoint + length / 2
    else:
        dx = length / (2 * np.sqrt(1 + slope**2))
        dy = slope * dx
        x1, y1 = midpoint - dx, midpoint - dy
        x2, y2 = midpoint + dx, midpoint + dy
    return [x1, y1, x2, y2]
def generate_representative_lines(groups, is_vertical):
    new_lines = []
    for group in groups:
        slopes = [compute_slope(line) for line in group['lines']]
        avg_slope = np.mean(slopes)
        if is_vertical:
            midpoints = [np.mean([line[0], line[2]]) for line in group['lines']]
        else:
            midpoints = [np.mean([line[1], line[3]]) for line in group['lines']]
        avg_midpoint = np.mean(midpoints)
        new_line = compute_new_line(avg_slope, avg_midpoint)
        new_lines.append(new_line)
    return new_lines
def find_intersections(lines):
    intersections = []
    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i < j:
                line1_geom = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
                line2_geom = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
                intersection = line1_geom.intersection(line2_geom)
                if not intersection.is_empty and isinstance(intersection, Point):
                    intersections.append(intersection)
    return intersections

image_path = "./box_coord/metal01.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_2 = image.copy()
image_3 = image.copy()


edges = cv2.Canny(image, threshold1 = 20, threshold2 = 50)
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=20, maxLineGap=20)
lines = [line[0] for line in lines]
print(f'lines: {lines}')

# 在image_3上畫霍夫直線
for line in lines:
    x1, y1, x2, y2 = line
    cv2.line(image_3, (x1, y1), (x2, y2), (255, 255, 255), 2)

angle_tolerance = 10
position_tolerance = 20

angle_groups = group_by_angle(lines, angle_tolerance)
print(f'angle_groups: {angle_groups}')

position_groups = []
for group in angle_groups:
    clusters = kmeans_cluster(group['lines'], 2)
    for cluster in clusters:
        position_groups.append({'lines': cluster})

print(f'grouped by position: {position_groups}')

vertical_groups = [position_groups[0], position_groups[1]]
horizontal_groups = [position_groups[2], position_groups[3]]
print(f'vertical_groups: {vertical_groups}')
print(f'horizontal_groups: {horizontal_groups}')

new_vertical_lines = generate_representative_lines(vertical_groups, is_vertical=True)
new_horizontal_lines = generate_representative_lines(horizontal_groups, is_vertical=False)
print(f'new_vertical_lines: {new_vertical_lines}')
print(f'new_horizontal_lines: {new_horizontal_lines}')

new_lines = new_vertical_lines + new_horizontal_lines
print(f'new lines: {new_lines}')

intersections = find_intersections(new_lines)
print(f'intersections: {intersections}')
print(type(intersections[0]))

# 取出交點座標
intersection_coords = [(point.x, point.y) for point in intersections]

# 將交點座標轉換為 numpy 數組
points = np.array(intersection_coords, dtype=np.float32)

# 使用 OpenCV 找到凸包
if len(points) >= 3:
    hull = cv2.convexHull(points)

# 畫圖
image_with_lines = image.copy()

# 畫出延長的線段
for line in new_lines:
    pt1 = (int(line[0]), int(line[1]))
    pt2 = (int(line[2]), int(line[3]))
    cv2.line(image_with_lines, pt1, pt2, (0, 255, 0), 1)

# 畫出凸面多邊形
if len(points) >= 3:
    hull = hull.astype(int)
    cv2.polylines(image_with_lines, [hull], isClosed=True, color=(0, 0, 255), thickness=2)

# 標記交點
for point in intersection_coords:
    cv2.circle(image_with_lines, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

# 計算並標記邊長
if len(points) >= 3:
    for i in range(len(hull)):
        pt1 = tuple(hull[i][0])
        pt2 = tuple(hull[(i + 1) % len(hull)][0])
        length = distance(pt1, pt2)
        midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(image_with_lines, f'{length:.2f}', midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


# 顯示結果
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(edges, cmap='gray')
axes[1].set_title('Canny Edges')
axes[2].imshow(image_3, cmap='gray')
axes[2].set_title('Lines Detected')
axes[3].imshow(image_with_lines, cmap='gray')
axes[3].set_title('Final Image')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()