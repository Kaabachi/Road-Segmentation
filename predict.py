import torch
from torchvision import transforms
import numpy as np
import math

PADDING = 40
PATCH_SIZE = 16
LARGE_PATCH_SIZE = 96
IMAGE_SIZE = 608
NUMBER_PATCHES_PER_IMAGE = 1444
        
def save_image(image, i):
    prediction_data_dir = './Datasets/predictions/'
    
    mask = image.clone().detach()
    
    img = transforms.ToPILImage()(mask)
    img.save(prediction_data_dir+"img" + str(i) + ".png","PNG")

def crop(image):
    return image[PADDING:PADDING+PATCH_SIZE, PADDING:PADDING+PATCH_SIZE]

def predict(model, dataloader):
    model.eval()
        
    tmp_img = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
    
    for ind_batch, sample_batched in enumerate(dataloader):
        batch_images = sample_batched["image"]
        output = model(batch_images)

        final = output[0][0].detach()
        final[final>0.5] = 1
        final[final<=0.5] = 0
        
        small_patch = crop(final)
        index = np.array([sample_batched['id'], sample_batched['x'], sample_batched['y']])
        
        image_number = index[0]
        
        start_x = index[1] * PATCH_SIZE
        end_x = start_x + PATCH_SIZE
        
        start_y = index[2] * PATCH_SIZE
        end_y = start_y + PATCH_SIZE 
        
        tmp_img[start_x:end_x, start_y:end_y] = small_patch
        
        if (index[1] == math.sqrt(NUMBER_PATCHES_PER_IMAGE) - 1) and (index[2] == math.sqrt(NUMBER_PATCHES_PER_IMAGE) - 1):
            save_image(tmp_img, index[0])
            tmp_img = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
        
        if ind_batch % 100 == 0:
                print(
                    "[Patch {}/{}]".format(
                        ind_batch, len(dataloader)
                    )
                )