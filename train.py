import torch.optim as optim

LEARNING_RATE = 0.0001

def train(model, X_tensor, Y_tensor, epochs, criterion, model_weights=None, batch_size=1):

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
#     assert batch_size divides X_tensor.size()[0]
    X = X_tensor.reshape(X_tensor.size()[0]//batch_size, batch_size, X_tensor.size()[1], X_tensor.size()[2], X_tensor.size()[3])
    Y = Y_tensor.reshape(Y_tensor.size()[0]//batch_size, batch_size, Y_tensor.size()[1], Y_tensor.size()[2])
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, X.size()[0]):
            image = X[i]
            groundtruth = Y[i]
            
            optimizer.zero_grad()
        
            output = model(image)
            
            print("out", output.size())
            
            loss = criterion(output[0],batch_groundtruth[0])
        
            loss.require_grad = True
            loss.backward()
            
            optimizer.step()

            if i % 10 == 0:
                print(
                    "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
                        epoch, i, X.size()[0] * batch_size, loss
                    )
                )
        
        
#         for ind_batch, sample_batched in enumerate(dataloader):
#             batch_images = sample_batched["image"]
#             batch_groundtruth = sample_batched["groundtruth"]
            
#             print(batch_images.size())
#             optimizer.zero_grad()

#             output = model(batch_images)
            
#             loss = criterion(
#                 output[0],
#                 batch_groundtruth[0]
#             )
            
#             loss.require_grad = True
#             loss.backward()
            
#             optimizer.step()

#             if ind_batch % 10 == 0:
#                 print(
#                     "[Epoch {}, Batch {}/{}]:  [Loss: {:03.2f}]".format(
#                         epoch, ind_batch, len(dataloader), loss
#                     )
#                 )