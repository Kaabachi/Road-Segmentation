import torch
from torchvision import transforms
import numpy as np


PADDING = 40
PATCH_SIZE = 16
LARGE_PATCH_SIZE = 96
IMAGE_SIZE = 608
NUMBER_PATCHES_PER_IMAGE = 1444

def reconstruct_images(result, indices):
    images = np.zeros((indices.shape[0]//NUMBER_PATCHES_PER_IMAGE, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
    
    for i in range(result.shape[0]):
        image_number = indices[i, 0]
        
        start_x = indices[i, 1] * PATCH_SIZE
        end_x = start_x + PATCH_SIZE
        
        start_y = indices[i, 2] * PATCH_SIZE
        end_y = start_y + PATCH_SIZE 
        
        images[image_number, start_x:end_x, start_y:end_y] = result[i]
        
    return images

def save_files(masks):
    prediction_data_dir = './Datasets/predictions/'
    
    mask = torch.tensor(masks, dtype=torch.float)
    
    for i in range(mask.size()[0]):
        print(mask[i])
        img = transforms.ToPILImage()(mask[i])
        img.save(prediction_data_dir+"img" + str(i) + ".png","PNG")

def crop(image):
    return image[PADDING:PADDING+PATCH_SIZE, PADDING:PADDING+PATCH_SIZE]

def predict(model, dataloader):
    model.eval()
    
    result = np.zeros((len(dataloader),PATCH_SIZE,PATCH_SIZE), dtype=np.int32)
    tmp = np.zeros((len(dataloader),LARGE_PATCH_SIZE,LARGE_PATCH_SIZE))
    indices = np.zeros((len(dataloader), 3), dtype=int)
    
    for ind_batch, sample_batched in enumerate(dataloader):
        batch_images = sample_batched["image"]
        output = model(batch_images)
        
        final = output[0].detach()
        final[final>0.5] = 1
        final[final<=0.5] = 0
                
        tmp[ind_batch] = np.array(final[0][0])
        result[ind_batch] = crop(tmp[ind_batch])
        indices[ind_batch] = np.array([sample_batched['id'], sample_batched['x'], sample_batched['y']])
        
        if ind_batch % 10 == 0:
                print(
                    "[Batch {}/{}]".format(
                        ind_batch, len(dataloader)
                    )
                )
        
    masks = reconstruct_images(result, indices)
    save_files(masks)
        
#     return result, indices, masks
        
        