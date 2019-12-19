import time
from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from config import (CRITERION, LARGE_BATCH_SIZE, NUMBER_PATCH_PER_IMAGE,
                    PADDING, PATCH_SIZE)
from config import TEST_BATCH_SIZE as BATCH_SIZE
from config import TRAIN_DATASET_DIR as DATASET_DIR
from config import TEST_MODEL as MODEL
from config import TEST_MODEL_WEIGTS as MODEL_WEIGHTS
from config import TRAIN_IMAGE_SIZE as IMAGE_SIZE
from datasets import RoadsDatasetValidation
from models.resnet import ResNet
from models.unet import UNet


def validate(model, dataloader, criterion, model_weights=None):
    """Calculate validation loss for a given model and a given dataset

        Args:
            model (nn.Module) : Model to be validated
            
            dataloader : The dataloader of the validation set
            
            criterion : Loss function used for validation
            
            checkpoints_dir : path for where to save the checkpoints
            
            model_weights : a dict containing the state of the weights with which to initialize our model

        """
    

    if model_weights is not None:
        model.load_state_dict(torch.load(model_weights))

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("CUDA available")
    else:
        print("NO CUDA")

    model.eval()
    global_loss = []
    for ind_batch, sample_batched in enumerate(dataloader):
        images = sample_batched["image"]
        groundtruths = sample_batched["groundtruth"]
        if cuda:
            images = images.to(device="cuda")
            groundtruths = groundtruths.to(device="cuda")

        output = model(images)

        loss = criterion(output, groundtruths)
        global_loss.append(loss)

    print("[Validation Loss: {:03.2f}]".format(mean(global_loss)))


if __name__ == "__main__":
    model = MODEL
    dataset = RoadsDatasetValidation(
        patch_size=PATCH_SIZE,
        large_patch_size=PATCH_SIZE,
        image_initial_size=IMAGE_SIZE,
        number_patch_per_image=NUMBER_PATCH_PER_IMAGE,
        root_dir=DATASET_DIR,
    )
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    validate(
        model=model,
        dataloader=dataloader,
        criterion=CRITERION,
        model_weights=MODEL_WEIGHTS,
    )
