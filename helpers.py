import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2

def pad(image, paddingh, paddingw, mirror=True):
    ''' Padd an image and fill the padded space with mirroring effect if `mirror` is set to True '''
    shape_image = list(image.shape)
    h, w = shape_image[:2]
    shape_image[0], shape_image[1] = h+paddingh*2, w+paddingw*2
    new_image = np.zeros(shape_image, np.uint8)
    new_image[paddingh:paddingh+h,paddingw:paddingw+w] = image

    # Fill in the padded regions with mirroring effect
    if mirror:
        new_image[0:paddingh] = cv2.flip(new_image[paddingh:2*paddingh],0)
        new_image[paddingh+h:] = cv2.flip(new_image[h:h+paddingh],0)
        new_image[:,0:paddingw] = cv2.flip(new_image[:,paddingw:2*paddingw],1)
        new_image[:,paddingw+w:] = cv2.flip(new_image[:,w:w+paddingw],1)
    return new_image

#get training set from directory, groundtruth and images
def get_train_set_bk(root_dir):
    
    root_dir = Path(root_dir)
    gt_dir = root_dir / "groundtruth"
    img_dir = root_dir / "images"
    
    df = pd.DataFrame(columns=["idx", "image", "groundtruth","image_name"])
    
    
    #save the id of each img
    idx = 0    
    img_name = [x.name for x in img_dir.glob("**/*.png") if x.is_file()]
    #traverse all images and save them in a dictionary with their name, id and groundtruth
    for name in img_name:
        image = np.array(Image.open(img_dir / name))
#         print(image.shape)
        groundtruth = np.array(Image.open(gt_dir / name))
        sample = pd.Series({"idx":idx,"image": image, "groundtruth": groundtruth,"image_name":name})
        df = df.append(sample, ignore_index=True)
        idx = idx+1

    return df

# get padding adequate to the size wanted
def pad_image(image,x_dim,y_dim):
    padded_image = pad(image,(x_dim-16)//2, (y_dim-16)//2)
#     padded_image = pad(image,x_dim//2, y_dim//2)
    return padded_image

# get patch_size x patch_size patches
def get_patches(image,padded_image,patch_size,x_dim,y_dim):
    patches=[]

    if len(image.shape)>2:
        sh,sw,sc = image.shape 
    else:
        sh,sw = image.shape
        sc = 1
        
    for j in range(patch_size//2,sw,patch_size):
        for i in range(patch_size//2, sh, patch_size):
            patches.append(padded_image[i:i+patch_size, j:j+patch_size])
    return patches

#get x_dim x y_dim patches centered in patch_size x patch_size
def get_large_patches(image,padded_image,patch_size,x_dim,y_dim):
    patches=[]
    
    if len(image.shape)>2:
        sh,sw,sc = image.shape 
    else:
        sh,sw = image.shape
        sc = 1    

    padding = (x_dim - patch_size)//2

    for j in range(padding,sw+padding,patch_size):
        for i in range(padding, sh+padding, patch_size):
            patches.append(padded_image[i-padding:i+y_dim-padding, j-padding:j+x_dim-padding])

    return patches

def get_train_dataset(path):
    x_dim= 96
    y_dim= 96
    patch_size = 16

    # load data
    data = get_train_set_bk(path)

    # pad images
    data["padded_img"] = data.apply(lambda row : pad_image(row["image"],x_dim,y_dim),axis = 1)
    tmp = []
    data.apply(lambda row : tmp.append(row["padded_img"]),axis = 1)
    images_kemlin = np.asarray(tmp)
#     images_kemlin = torch.from_numpy(images_kemlin)
    
    data["padded_groundtruth"] = data.apply(lambda row : pad_image(row["groundtruth"],x_dim,y_dim),axis = 1)
    tmp = []
    data.apply(lambda row : tmp.append(row["padded_groundtruth"]),axis = 1)
    groundtruth_kemlin = np.asarray(tmp)
#     groundtruth_kemlin = torch.from_numpy(grountruth_kemlin)
    
    
#     data["padded_groundtruth"] = data.apply(lambda row : pad_image(row["groundtruth"],x_dim,y_dim),axis = 1)
    
    groundtruth = data.apply(lambda row : get_large_patches(row["groundtruth"], row["padded_groundtruth"], patch_size,x_dim,y_dim), axis=1).tolist()
    groundtruth = np.asarray(groundtruth)
    groundtruth = torch.from_numpy(groundtruth)
    groundtruth = groundtruth.reshape(groundtruth.size()[0]*groundtruth.size()[1],groundtruth.size()[2], groundtruth.size()[3])
    
    images = data.apply(lambda row :get_large_patches(row["image"],row["padded_img"],patch_size,x_dim,y_dim),axis = 1).tolist()
    images = np.asarray(images)
    images = torch.from_numpy(images)
    images = images.permute(0,1,4,2,3)
    images = images.reshape(images.size()[0]*images.size()[1],images.size()[2], images.size()[3], images.size()[4])
    
    
    return images, groundtruth, images_kemlin, groundtruth_kemlin