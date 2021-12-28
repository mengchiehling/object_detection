import os

from algorithms.settings import INFERENCE_MODEL
from algorithms.io.aws_connection_definition import download_aws_folder, download_aws_file
from algorithms.io.path_definition import get_project_dir

if __name__ == "__main__":

    download_aws_folder(f"data/app/models/YOLO/{INFERENCE_MODEL['YOLO']}/",
                        local_folder=f"{get_project_dir()}/data/app/models/YOLO/{INFERENCE_MODEL['YOLO']}/")

    local_folder = f"{get_project_dir()}/data/app/models/Mask_RCNN/{INFERENCE_MODEL['Mask_RCNN']}"

    if not os.path.isdir(local_folder):
        os.makedirs(local_folder)

    download_aws_file(f"data/app/models/Mask_RCNN/{INFERENCE_MODEL['Mask_RCNN']}/best_model.h5",
                      local_file_name=f"{local_folder}/best_model.h5")
