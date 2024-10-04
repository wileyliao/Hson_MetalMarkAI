import os


def generate_file_path_and_name(db_path, api_product_code, api_date_info, api_time_info):

    absolute_path_to_db = os.path.join(db_path, api_product_code, api_date_info)

    if not os.path.exists(absolute_path_to_db):
        os.makedirs(absolute_path_to_db)

    db_image_file_name = f'{api_time_info}.png'

    return absolute_path_to_db, db_image_file_name
