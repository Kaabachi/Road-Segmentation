import math

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from config import LARGE_BATCH_SIZE, LARGE_PATCH_SIZE, PADDING, PATCH_SIZE
from config import TEST_BATCH_SIZE as BATCH_SIZE
from config import TEST_DATASET_DIR as DATASET_DIR
from config import TEST_IMAGE_SIZE as IMAGE_SIZE
from config import TEST_MODEL as MODEL
from config import TEST_MODEL_WEIGTS as MODEL_WEIGHTS
from config import TEST_NUMBER_PATCH_PER_IMAGE
from datasets import RoadsDatasetTest


def save_image(image, i):
    prediction_data_dir = "./Datasets/predictions/"

    mask = image.clone().detach().cpu()

    img = transforms.ToPILImage()(mask)
    img.save(prediction_data_dir + "img" + str(i) + ".png", "PNG")


def crop(image):
    return image[PADDING : PADDING + PATCH_SIZE, PADDING : PADDING + PATCH_SIZE]


def predict(model, dataloader, model_weights=None):

    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights))

    model.eval()

    tmp_img = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("CUDA is available")
    else:
        print("CUDA is NOT available")

    for ind_batch, sample_batched in enumerate(dataloader):
        batch_images = sample_batched["image"]
        if cuda:
            batch_images = batch_images.to(device="cuda")
        with torch.no_grad():
            output = model(batch_images)

        final = output[0][0].clone().detach().cpu()
        final[final > 0.5] = 1
        final[final <= 0.5] = 0

        small_patch = crop(final)
        index = np.array(
            [sample_batched["id"], sample_batched["x"], sample_batched["y"]]
        )

        image_number = index[0]

        start_x = index[1] * PATCH_SIZE
        end_x = start_x + PATCH_SIZE

        start_y = index[2] * PATCH_SIZE
        end_y = start_y + PATCH_SIZE

        tmp_img[start_x:end_x, start_y:end_y] = small_patch

        if (index[1] == math.sqrt(TEST_NUMBER_PATCH_PER_IMAGE) - 1) and (
            index[2] == math.sqrt(TEST_NUMBER_PATCH_PER_IMAGE) - 1
        ):
            save_image(tmp_img, index[0])
            tmp_img = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)

        if ind_batch % 100 == 0:
            print("[Patch {}/{}]".format(ind_batch, len(dataloader)))


if __name__ == "__main__":
    model = MODEL
    dataset = RoadsDatasetTest(
        patch_size=PATCH_SIZE,
        large_patch_size=LARGE_PATCH_SIZE,
        number_patch_per_image=TEST_NUMBER_PATCH_PER_IMAGE,
        image_initial_size=IMAGE_SIZE,
        root_dir=DATASET_DIR,
    )
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    predict(model=model, dataloader=dataloader, model_weights=MODEL_WEIGHTS)
