import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
from transformations import *
from helpers import pad_image

# IMAGE_INITIAL_SIZE = 400
# PATCH_SIZE = 16
# LARGE_PATCH_SIZE = 96
# PADDING = (LARGE_PATCH_SIZE - PATCH_SIZE)//2 #40

# NUMBER_PATCHES_PER_ROW = IMAGE_INITIAL_SIZE // PATCH_SIZE #25
# NUMBER_PATCHES_PER_IMAGE = NUMBER_PATCHES_PER_ROW * NUMBER_PATCHES_PER_ROW #25

def extract_images(img_names, root_dir=None):
    images = []
    
    if(root_dir != None):
        for i in range(len(img_names)):
            name = img_names[i]
            image = Image.open(root_dir / name)

            images.append(image)
    else:
        for i in range(len(img_names)):
            name = img_names[i]
            image = Image.open(name)

            images.append(image)
        
    return images

class RoadsDatasetTrain(Dataset):
    """Road segmentation datset"""

    def __init__(self,patch_size, large_patch_size, number_patch_per_image, image_initial_size, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_dir = self.root_dir / "images"
        self.gt_dir = self.root_dir / "groundtruth"
        self.img_names = [x.name for x in self.img_dir.glob("**/*.png") if x.is_file()]
        self.patch_size = patch_size
        self.images = extract_images(self.img_names, self.img_dir)
        self.groundtruths = extract_images(self.img_names, self.gt_dir)
        self.large_patch_size = large_patch_size
        self.number_patch_per_image = number_patch_per_image
        self.image_initial_size = image_initial_size

    def __len__(self):
        return self.number_patch_per_image * len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        l_p_s = self.large_patch_size
        n_p_p_i = self.number_patch_per_image
        image_index = idx//n_p_p_i
        patch_index = idx%n_p_p_i
        
        img_name = self.img_names[image_index]
        image = self.images[image_index]
        groundtruth = self.groundtruths[image_index]
        
        padded_image = pad_image(np.array(image),l_p_s,l_p_s)
        padded_groundtruth = pad_image(np.array(groundtruth), l_p_s, l_p_s)
        
        small_image, small_groundtruth = self._get_patch(padded_image, padded_groundtruth, patch_index)
        
        sample = {"image": small_image, "groundtruth": small_groundtruth}

        if self.transform:
            sample = self.transform(sample)
        else:
            transformation = transforms.Compose(
                [ToTensor()]
            )
            sample = transformation(sample)

        return sample
    
    def _get_patch(self, image, groundtruth, patch_number):
        n_s_p_p_i = self.image_initial_size // self.patch_size
        
        p_s = self.patch_size
        l_p_s = self.large_patch_size
        
        padding = (self.large_patch_size - self.patch_size) // 2
        
        
        y = ((patch_number % n_s_p_p_i) * p_s)+padding
        x = ((patch_number // n_s_p_p_i) * p_s)+padding
    
        image_patch = image[x-padding:x+l_p_s-padding, y-padding:y+l_p_s-padding]
    
        groundtruth_patch = groundtruth[x-padding:x+l_p_s-padding, y-padding:y+l_p_s-padding]
    
        return image_patch, groundtruth_patch


class RoadsDatasetTest(Dataset):
    """Road segmentation dataset for test time"""

    def __init__(self,patch_size, large_patch_size, number_patch_per_image, image_initial_size,root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_names = [str(x) for x in self.root_dir.glob("**/*.png") if x.is_file()]
        self.patch_size = patch_size
        self.images = extract_images(self.img_names)
        self.large_patch_size = large_patch_size
        self.number_patch_per_image = number_patch_per_image
        self.image_initial_size = image_initial_size

    def __len__(self):
        return self.number_patch_per_image * len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        l_p_s = self.large_patch_size
        n_p_p_i = self.number_patch_per_image
        image_index = idx//n_p_p_i
        patch_index = idx%n_p_p_i
        
        img_name = self.img_names[image_index]
        
        image = self.images[image_index]
        
        padded_image = pad_image(np.array(image),l_p_s,l_p_s)
        
        small_image, x, y = self._get_patch(padded_image, patch_index)
        
        sample = small_image
        
        if self.transform:
            sample = self.transform(image)
        else:
            transformation = transforms.Compose(
                [
                    transforms.ToTensor()
                ]
            )
            sample = transformation(sample)
            
        sample = {"id": image_index, "x" : x, "y" : y, "image": sample}

        return sample
    
    def _get_patch(self, image, patch_number):
        n_s_p_p_i = self.image_initial_size // self.patch_size
        
        p_s = self.patch_size
        l_p_s = self.large_patch_size
        
        padding = (self.large_patch_size - self.patch_size) // 2
        
        y = ((patch_number % n_s_p_p_i) * p_s)+padding
        x = ((patch_number // n_s_p_p_i) * p_s)+padding
    
        image_patch = image[x-padding:x+l_p_s-padding, y-padding:y+l_p_s-padding]
        
        x = (x-padding)//p_s
        y = (y-padding)//p_s
    
        return image_patch, x, y
