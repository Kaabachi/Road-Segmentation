import numpy as np
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torchvision import transforms
import models
import random
from datasets import RoadsDatasetTest
from transformations import (
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    ToTensor,
    Normalize,
    RandomRotation,
)


CHECKPOINTS_DIR = "./checkpoints"
MODEL_WEIGHTS = "20191213-104938_FCN_epoch_15_loss_0.234.pt"  # put weights
DATA_DIR = "./Datasets/test_set_images"
OUTPUT_DIR = "./output"
OUT_SIZE = 608
MODEL = models.fcn_resnet50
BATCH_SIZE = 2
# we have 2 classes
OUTPUT_CHANNELS = 2


def output_mask(output, image_names, output_dir=OUTPUT_DIR):
    de_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.ToPILImage(mode='RGB'),
            # transforms.ToTensor(),
            # transforms.Normalize(
            # mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
            # ),
            # transforms.Normalize(mean=[-0.485, -0.456, -0.456], std=[1.0, 1.0, 1.0]),
            # transforms.ToPILImage(),
            #transforms.Resize(size=OUT_SIZE),
        ]
    )
    se_transform = transforms.Compose([

            transforms.ToTensor(),
            #transforms.Normalize(
            #mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
            #),
            #transforms.Normalize(mean=[-0.485, -0.456, -0.456], std=[1.0, 1.0, 1.0]),
            transforms.ToPILImage(),
            transforms.Resize(size=OUT_SIZE),
    ])
    for mask, image_name in zip(output, image_names):
        mask = de_transform(mask)
        mask = mask.convert('RGB')
        mask = se_transform(mask)
        output_path = Path(output_dir) / image_name
        mask.save(output_path)
        print(f"saved {image_name} in {output_dir}")


def eval(model, dataloader, model_weights=None):

    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights))

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("GPU available")
    else:
        print("NO GPU")

    for ind_batch, sample_batched in enumerate(dataloader):
        batch_images = sample_batched["image"]
        image_names = sample_batched["img_name"]
        if cuda:
            batch_images = batch_images.to(device="cuda")
        with torch.no_grad():
            output = model(batch_images)["out"].clone().detach().cpu().clamp(0.0, 1.0)
            output_mask(output=output, image_names=image_names, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    model = MODEL
    dataset = RoadsDatasetTest(root_dir=DATA_DIR)
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    eval(
        model=model,
        dataloader=dataloader,
        model_weights=str(Path(CHECKPOINTS_DIR) / MODEL_WEIGHTS),
    )

