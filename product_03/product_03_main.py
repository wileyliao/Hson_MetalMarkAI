import random
import time


def product_03_main(image_path, product_03_model, product_03_rows, product_03_columns):
    try:
        start_time = time.time()

        random_fail_row = random.randint(0, product_03_rows - 1)
        random_fail_col = random.randint(0, product_03_columns - 1)

        classification_results = {}
        for row in range(product_03_rows):
            for column in range(product_03_columns):
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
