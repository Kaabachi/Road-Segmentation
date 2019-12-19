import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2
import re

# pad an image and fill the padded space with mirroring effect if `mirror` is set to True
def pad(image, paddingh, paddingw, mirror=True):
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

# Take an predictions and remove noise from them using morphological closing then opening.
def post_processing():
    for i in range(1, 51):
        #read images
        image_filename = "Predictions/img" +  str(i) + ".png"
        img = cv2.imread(image_filename)
        
        #kernel to use in convolution
        kernel = np.ones((10,10),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        #replace old image with new one
        cv2.imwrite(image_filename, img );



# get padding adequate to the size wanted
def pad_image(image,x_dim,y_dim, pad):
    padded_image = pad(image,(x_dim-16)//2, (y_dim-16)//2)
#     padded_image = pad(image,x_dim//2, y_dim//2)
    return padded_image

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

# Functions used to sort image names when loading test dataset so that the loading is done in order

def to_int(text):
    '''
    Transform text to int if it is digit
    '''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    '''
    return [ to_int(c) for c in re.split(r'(\d+)', text) ]

