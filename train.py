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
from datasets import RoadsDatasetTrain
from transformations import (
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    ToTensor,
    Normalize,
    RandomRotation,
)


# temp values
DATA_DIR = "./Datasets/training"
CHECKPOINTS_DIR = "./checkpoints"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose(
    [
        Resize(224),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ColorJitter(),
        ToTensor(),
        Normalize(mean=MEAN, std=STD),
    ]
)
MODEL = models.fcn_resnet50
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 2
CRITERION = nn.BCELoss()
# we have 2 classes
OUTPUT_CHANNELS = 2


def save_model(model, epoch, loss, save_dir):
   model_name =  model.__class__.__name__
   timestr = time.strftime("%Y%m%d-%H%M%S")
   file_name = f'{timestr}_{model_name}_epoch_{epoch}_loss_{loss}.pt'
   Path(save_dir).mkdir(exist_ok=True)
   file_path = Path(save_dir) / file_name
   torch.save(model.state_dict(), str(file_path))


def train(model, dataloader, epochs, criterion, model_weights=None):

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("GPU available")
    else:
        print("NO GPU")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(epochs):
        for ind_batch, sample_batched in enumerate(dataloader):
            batch_images = sample_batched["image"]
            batch_groundtruth = sample_batched["groundtruth"]
            if cuda:
                batch_images = batch_images.to(device="cuda")
                batch_groundtruth = batch_groundtruth.to(device="cuda")
            optimizer.zero_grad()

            output = model(batch_images)["out"]
            #output_predictions = (
                #output.argmax(1).type(torch.float).clone().detach().requires_grad_(True)
            #)
            groundtruth_predictions = (batch_groundtruth > 0.5).float()

            # TODO: check if this loss is working
            loss = criterion(output.clamp(0.0, 1.0).view(-1), batch_groundtruth.view(-1))
            # loss.require_grad = True
            loss.backward()
            optimizer.step()

            if ind_batch % 10 == 0:
                print(
                    f"[Epoch {epoch}, Batch {ind_batch:02d}/{len(dataloader)}]:  [Loss: {loss.item():03.2f}]"
                )

        if epoch % 5 == 0:
            save_model(model, epoch, loss.item(), CHECKPOINTS_DIR)
            print(f'model saved to {str(CHECKPOINTS_DIR)}')


if __name__ == "__main__":
    # TODO: model
    model = MODEL
    dataset = RoadsDatasetTrain(root_dir=DATA_DIR, transform=TRANSFORM)
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(model=model, dataloader=dataloader, epochs=EPOCHS, criterion=CRITERION)
