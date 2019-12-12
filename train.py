import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import models
from datasets import RoadsDatasetTrain


# temp values
MODEL = models.fcn_resnet101
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
CRITERION = nn.BCELoss()
# we have 2 classes
OUTPUT_CHANNELS = 2


def train(model, dataloader, epochs, criterion, model_weights=None):

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.to(device="cuda")
        print("GPU available")
    else:
        print("NO GPU")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        for ind_batch, sample_batched in enumerate(dataloader):
            batch_images = sample_batched["image"]
            batch_groundtruth = sample_batched["groundtruth"]
            if cuda:
                batch_images = batch_images.to(device="cuda")
                batch_groundtruth = batch_groundtruth.to(device="cuda")
            optimizer.zero_grad()

            output = model(batch_images)["out"]
            output_predictions = torch.tensor(
                output.argmax(1).type(torch.float), requires_grad=True
            )

            # TODO: check if this loss is working
            loss = criterion(output_predictions.view(-1), batch_groundtruth.view(-1))
            # loss.require_grad = True
            loss.backward()
            optimizer.step()

            if ind_batch % 10 == 0:
                print(
                    f"[Epoch {epoch}, Batch {ind_batch:02d}/{len(dataloader)}]:  [Loss: {loss.item():03.2f}]"
                )


if __name__ == "__main__":
    # TODO: model
    model = MODEL
    data_dir = "./Datasets/training"
    dataset = RoadsDatasetTrain(root_dir=data_dir)
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(model=model, dataloader=dataloader, epochs=EPOCHS, criterion=CRITERION)
