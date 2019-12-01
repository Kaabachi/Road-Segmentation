# file to prepare the dataset into a detectron2 compatible format
# see https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

import json
import os
from PIL import Image
from pathlib import Path



def get_img_size(img_path):
    """
    returns height and width of img_path
    """
    image = Image.open(img_path)
    width, height = image.size
    return width, height


if __name__ == "__main__":
    print("hello")
    DATASET = Path("./Datasets/training")
    IMAGES = DATASET / "images"
    GROUNDTRUTH = DATASET / "groundtruth"
    IMG_FORMAT = ".png"
    json_dicts = DATASET / 'json_dicts'
    json_dicts.mkdir(exist_ok=True)
    for img_path in IMAGES.glob("**/*"+IMG_FORMAT):
        filename = img_path
        print(f"Treating {filename.name}")
        sem_seg_file_name = GROUNDTRUTH / img_path.name
        width, height = get_img_size(img_path)
        img_id = img_path.name.split("_")[1].split(".")[0]
        img_dict = {
            "filename": str(filename),
            "sem_seg_file_name": str(sem_seg_file_name),
            "height": height,
            "width": width,
            "img_id": img_id,
        }
        json_path = json_dicts / (filename.stem + ".json")
        print(f"Writing {json_path.name}")
        with open(json_path, 'w') as json_outfile:
            json.dump(img_dict, json_outfile)