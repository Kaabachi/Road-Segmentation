import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from datasets import RoadsDatasetTrain


# temp values
EPOCHS = 100
LEARNING_RATE = 0.0001
BATCH_SIZE = 100
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

    optimizer = optim.adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        for ind_batch, sample_batched in enumerate(dataloader):
            if cuda:
                samples = samples.to(device="cuda")
            optimizer.zero_grad()

            output = model(sample_batched["image"])

            # TODO: check if this loss is working
            loss = criterion(
                output.permute(0, 2, 3, 1).contiguous.view(-1, OUTPUT_CHANNELS),
                sample_batched["groundtruth"].view(-1),
            )
            loss.backward()
            optimizer.step()

            if ind_batch % 10 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, ind_batch, len(dataloader), loss.data[0]
                    )
                )


if __name__ == "__main__":
    # TODO: model
    model = None
    data_dir = "./Datasets/training"
    dataset = RoadsDatasetTrain(root_dir=data_dir)
    dataloader = data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    train(model=model, dataloader=dataloader, epochs=EPOCHS, criterion=CRITERION)
