import torch
from torchvision import transforms

def predict(model, dataloader):
    model.eval()
    prediction_data_dir = './Datasets/predictions/'
    
    for ind_batch, sample_batched in enumerate(dataloader):
        batch_images = sample_batched["image"]
        output = model(batch_images)
        
        final = output[0][0]
        final[final>0.5] = 1
        final[final<=0.5] = 0
        
        img = transforms.ToPILImage()(final)
        img.save(prediction_data_dir+"img" + str(ind_batch) + ".png","PNG")