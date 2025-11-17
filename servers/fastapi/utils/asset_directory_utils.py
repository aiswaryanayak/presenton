import os
from utils.get_env import get_app_data_directory_env

def get_images_directory():
    base = get_app_data_directory_env()
    images_directory = os.path.join(base, "images")
    os.makedirs(images_directory, exist_ok=True)
    return images_directory

def get_exports_directory():
    base = get_app_data_directory_env()
    export_directory = os.path.join(base, "exports")
    os.makedirs(export_directory, exist_ok=True)
    return export_directory

def get_uploads_directory():
    base = get_app_data_directory_env()
    uploads_directory = os.path.join(base, "uploads")
    os.makedirs(uploads_directory, exist_ok=True)
    return uploads_directory
