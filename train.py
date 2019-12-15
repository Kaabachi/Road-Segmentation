import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from datasets import RoadsDatasetTrain
from models.unet import UNet

LEARNING_RATE = 0.0001
MODEL_NAME = "unet"
CHECKPOINTS_DIR = "./checkpoints"
BATCH_SIZE = 1
CRITERION = nn.BCELoss()
EPOCHS = 100
PATCH_SIZE = 16
LARGE_PATCH_SIZE = 400
TRAIN_IMAGE_INITIAL_SIZE = 400
NUMBER_PACH_PER_IMAGE = int((TRAIN_IMAGE_INITIAL_SIZE / PATCH_SIZE)**2)
DATASET_DIR = "./Datasets/training"


def save_model(model, epoch, loss, save_dir):
    model_name = MODEL_NAME
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"{timestr}_{model_name}_epoch_{epoch}_loss_{loss:03.3f}.pt"
    Path(save_dir).mkdir(exist_ok=True)
    file_path = Path(save_dir) / file_name
    torch.save(model.state_dict(), str(file_path))


def train(
    model,
    dataloader,
    epochs,
    criterion,
    model_weights=None,
    checkpoints_dir=CHECKPOINTS_DIR,
):

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("CUDA available")
    else:
        print("NO CUDA")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        for ind_batch, sample_batched in enumerate(dataloader):
            images = sample_batched["image"]
            groundtruths = sample_batched["groundtruth"]
            if cuda:
                images = images.to(device="cuda")
                groundtruths = groundtruths.to(device="cuda")

            optimizer.zero_grad()

            output = model(images)

            loss = criterion(output, groundtruths)

            loss.require_grad = True
            loss.backward()

            optimizer.step()

            if ind_batch % 10 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, ind_batch, len(dataloader), loss
                    )
                )
        if epoch % 5 == 0:
            save_model(
                model=model, epoch=epoch, loss=loss.item(), save_dir=checkpoints_dir
            )
            print(f"model saved to {str(checkpoints_dir)}")


if __name__ == "__main__":
    model = UNet()
    dataset = RoadsDatasetTrain(
        patch_size=PATCH_SIZE,
        large_patch_size=LARGE_PATCH_SIZE,
        image_initial_size=TRAIN_IMAGE_INITIAL_SIZE,
        number_patch_per_image=NUMBER_PACH_PER_IMAGE,
        root_dir=DATASET_DIR,
    )
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(
        model=model,
        dataloader=dataloader,
        epochs=EPOCHS,
        criterion=CRITERION,
        checkpoints_dir=CHECKPOINTS_DIR,
    )