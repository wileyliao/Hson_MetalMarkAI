import os


def generate_file_path_and_name(db_path, api_product_code, api_date_info, api_time_info):

    absolute_path_to_db = os.path.join(db_path, api_product_code, api_date_info)

    os.makedirs(absolute_path_to_db, exist_ok=True)

    db_image_file_name = api_time_info

    return absolute_path_to_db, db_image_file_name
