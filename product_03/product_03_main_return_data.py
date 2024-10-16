def calculate_matrix_positions(boxes):
    """
    根據產品框座標計算它們在 2x2 矩陣中的相對位置。

    Args:
        boxes: 每個產品的框座標列表，格式為 [(x1, y1, x2, y2), ...]

    Returns:
        matrix_positions: 字典，記錄每個產品的矩陣位置。
    """
    # 按照 y1 進行排序，找到上排和下排
    sorted_by_y = sorted(enumerate(boxes), key=lambda item: item[1][1])  # 以 y1 作為排序依據
    top_row = sorted_by_y[:2]  # y1 最小的兩個在上排
    bottom_row = sorted_by_y[2:]  # y1 最大的兩個在下排

    # 分別對上排和下排按照 x1 進行排序，確定左右列
    top_row_sorted = sorted(top_row, key=lambda item: item[1][0])  # 以 x1 作為排序依據
    bottom_row_sorted = sorted(bottom_row, key=lambda item: item[1][0])  # 以 x1 作為排序依據

    # 根據排序結果確定位置
    matrix_positions = {}
    matrix_positions[top_row_sorted[0][0]] = (0, 1)  # 左上角
    matrix_positions[top_row_sorted[1][0]] = (1, 1)  # 右上角
    matrix_positions[bottom_row_sorted[0][0]] = (0, 0)  # 左下角
    matrix_positions[bottom_row_sorted[1][0]] = (1, 0)  # 右下角

    return matrix_positions


def check_point_and_door_relation(results_dict, boxes):
    """
    檢查 point 和 door 之間的相對位置，返回判斷結果。

    Args:
        results_dict: 包含每張裁剪後圖片檢測結果的字典，格式為 {id: result}
        boxes: 每個裁剪後影像的框座標列表。

    Returns:
        relations: 字典，記錄每個產品的 pass/fail 判斷
    """
    relations = {}

    for idx, (box, result) in enumerate(zip(boxes, results_dict.values())):
        door_coords = None
        point_coords = None

        # 從檢測結果中找到 door 和 point 的位置
        for box_result, cls in zip(result[0].boxes.xyxy, result[0].boxes.cls):
            x1, y1, x2, y2 = box_result
            if cls == 0:  # Door
                door_coords = (x1, y1, x2, y2)
            elif cls == 1:  # Point
                point_coords = (x1, y1, x2, y2)

        if door_coords and point_coords:
            # 判斷 point 是否在 door 的右上方
            door_x_center = (door_coords[0] + door_coords[2]) / 2
            point_x_center = (point_coords[0] + point_coords[2]) / 2
            point_y_center = (point_coords[1] + point_coords[3]) / 2

            # 判斷 point 是否在 door 的右側，且在上方
            if point_x_center > door_x_center and point_y_center < door_coords[1]:
                relations[idx] = {"box": box, "relation": "pass"}
            else:
                relations[idx] = {"box": box, "relation": "fail"}
        else:
            # 如果沒有偵測到其中一個，則標記為 "fail"
            relations[idx] = {"box": box, "relation": "fail"}

    return relations


def calculate_matrix_positions_and_relations(boxes, results_dict):
    """
    根據產品框座標計算它們在 2x2 矩陣中的相對位置，並判斷 point 和 door 的相對位置關係。

    Args:
        boxes: 每個產品的框座標列表，格式為 [(x1, y1, x2, y2), ...]
        results_dict: 包含每張裁剪後圖片檢測結果的字典，格式為 {id: result}

    Returns:
        matrix_relations: 字典，記錄每個產品的矩陣位置和 pass/fail 判斷。
                          格式為 {(row, col): "pass" or "fail"}
    """
    # 步驟 1：計算矩陣位置
    matrix_positions = calculate_matrix_positions(boxes)

    # 步驟 2：判斷 point 和 door 的相對位置
    relations = check_point_and_door_relation(results_dict, boxes)

    # 步驟 3：合併矩陣位置和判斷結果
    matrix_relations = {}
    for idx, position in matrix_positions.items():
        # 使用矩陣位置作為鍵，將相應的判斷結果（pass/fail）存入字典
        relation_result = relations[idx]["relation"]
        matrix_relations[position] = relation_result

    return matrix_relations
