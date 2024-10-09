# algorithm:
# 圖像預處理(縮小、轉成gpu張量)
# 檢測物體(定位銅板位置)
# 從原始影像裁剪出每一塊銅板
# 二次檢測並標記結果

import random
import time


def product_03_main(image_path, product_03_model, product_03_rows, product_03_columns):
    try:
        start_time = time.time()

        random_fail_row = random.randint(0, product_03_rows)
        random_fail_col = random.randint(0, product_03_columns)
        print(product_03_rows)

        classification_results = {}
        for row in range(product_03_rows + 1):
            for column in range(product_03_columns + 1):
                if (row, column) == (random_fail_row, random_fail_col):
                    classification_results[(row, column)] = 'fail'
                else:
                    classification_results[(row, column)] = 'pass'



        time.sleep(random.uniform(0.5, 1.5))
        end_time = time.time()
        execution_time = end_time - start_time
        return classification_results, execution_time

    except Exception as e:
        raise e
