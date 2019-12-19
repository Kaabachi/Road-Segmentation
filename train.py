import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from config import CRITERION as CRITERION
from config import EPOCHS as EPOCHS
from config import (LARGE_PATCH_SIZE, LEARNING_RATE, NUMBER_PATCH_PER_IMAGE,
                    PATCH_SIZE, SAVE_MODEL_EVERY_X_EPOCH)
from config import TRAIN_BATCH_SIZE as BATCH_SIZE
from config import TRAIN_CHECKPOINTS_DIR as CHECKPOINTS_DIR
from config import TRAIN_DATASET_DIR as DATASET_DIR
from config import TRAIN_IMAGE_INITIAL_SIZE
from config import TRAIN_MODEL as MODEL
from datasets import RoadsDatasetTrain
from models.resnet import ResNet
from models.unet import UNet


def save_model(model, epoch=None, loss=None, save_dir=None, specific_name=None):
    """Saves a checkpoint of the model state

        Args:
            model (nn.Module) : Model to be saved

            epoch : The epoch of this checkpoint
            
            loss : The loss of the last batch of this checkpoint
            
            save_dir : path for where to save the checkpoint
            
            specifi_name : a specific name to give to the checkpoint

        """
    if epoch and loss and save_dir and specific_name:
        model_name = model.model_name
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name = f"{timestr}_{model_name}_epoch_{epoch}_loss_{loss:03.3f}.pt"
        Path(save_dir).mkdir(exist_ok=True)
        file_path = Path(save_dir) / file_name
        torch.save(model.state_dict(), str(file_path))
    elif save_dir and specific_name:
        file_path = Path(save_dir) / specific_name
        torch.save(model.state_dict(), str(file_path))

def train(
    model,
    dataloader,
    epochs,
    criterion,
    model_weights=None,
    checkpoints_dir=CHECKPOINTS_DIR,
    last_checkpoint=None,
):
    """Trains the model

        Args:
            model (nn.Module) : Model to be trained
            
            dataloader : The dataloader of the training set

            epochs (int) : Number of epochs to train the model
            
            criterion : Loss function for the training
            
            checkpoints_dir : path for where to save the checkpoints
            
            model_weights : a dict containing the state of the weights with which to initialize our model

        """

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

            if ind_batch % 100 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, ind_batch, len(dataloader), loss
                    )
                )

        if SAVE_MODEL_EVERY_X_EPOCH and (epoch % SAVE_MODEL_EVERY_X_EPOCH == 0):
            save_model(
                model=model, epoch=epoch, loss=loss.item(), save_dir=checkpoints_dir
            )
#             Every SAVE_MODEL_EVERY_X_EPOCH, we make a checkpoint of our model
            print(f"model saved to {str(checkpoints_dir)}")
    save_model(
        model=model,
        epoch=None,
        loss=None,
        save_dir=checkpoints_dir,
        specific_name=last_checkpoint,
    )



if __name__ == "__main__":
    model = MODEL
    dataset = RoadsDatasetTrain(
        patch_size=PATCH_SIZE,
        large_patch_size=LARGE_PATCH_SIZE,
        image_initial_size=TRAIN_IMAGE_INITIAL_SIZE,
        number_patch_per_image=NUMBER_PATCH_PER_IMAGE,
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
