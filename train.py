import torch.optim as optim

LEARNING_RATE = 0.0001

def train(model, dataloader, epochs, criterion, model_weights=None):

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        for ind_batch, sample_batched in enumerate(dataloader):
            batch_images = sample_batched["image"]
            batch_groundtruth = sample_batched["groundtruth"]
            
            optimizer.zero_grad()

            output = model(batch_images)
            
            loss = criterion(
                output,
                batch_groundtruth
            )
            
            loss.require_grad = True
            loss.backward()
            
            optimizer.step()

            if ind_batch % 10 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, ind_batch, len(dataloader), loss
                    )
                )