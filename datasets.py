import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from PIL import Image
import transformations
from helpers import pad_image, to_int, natural_keys
from torchvision.transforms import functional as F
import math


class RoadsDatasetTrain(Dataset):
    """Road segmentation datset"""

    def __init__(self,patch_size, large_patch_size, number_patch_per_image, image_initial_size, root_dir):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.gt_dir = self.root_dir / "groundtruth"
        
        self.img_names = [x.name for x in self.img_dir.glob("**/*.png") if x.is_file()]
        # Sort images to in a human readable way
        self.img_names.sort(key=natural_keys)
        

        self.large_patch_size = large_patch_size
        self.number_patch_per_image = number_patch_per_image
        self.image_initial_size = image_initial_size
        self.patch_size = patch_size
        
        self.transforms = self._get_transforms()
        self.images, self.groundtruths = self._extract_images()

    def __len__(self):
        return self.number_patch_per_image * len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        l_p_s = self.large_patch_size
        n_p_p_i = self.number_patch_per_image
        n_s_p_p_i = self.image_initial_size // self.patch_size
        p_s = self.patch_size
        l_p_s = self.large_patch_size
        
        image_index = idx//n_p_p_i
        patch_index = idx%n_p_p_i
        
        padding = (l_p_s - p_s) // 2
        
        y = ((patch_index % n_s_p_p_i) * p_s)+padding
        x = ((patch_index // n_s_p_p_i) * p_s)+padding

        image = self.images[image_index]
        groundtruth = self.groundtruths[image_index]
        
        small_image = F.crop(image,x-padding,y-padding,l_p_s,l_p_s)
        small_groundtruth = F.crop(groundtruth,x-padding,y-padding,l_p_s,l_p_s)
        
        sample = {"image": small_image, "groundtruth": small_groundtruth}

        transformation = transforms.Compose(
            [transformations.ToTensor()]
        )
        
        sample = transformation(sample)

        return sample
    
    def _extract_images(self):
        images = []
        groundtruths = []
  
        transforms = self.transforms
        
        for i in range(len(self.img_names)):
            name = self.img_names[i]
            image = Image.open(self.img_dir / name)
            groundtruth = Image.open(self.gt_dir / name)
            
            sample = {'image':image, 'groundtruth':groundtruth}
            
            for j in range(len(transforms)):
                transformed_image = transforms[j](sample)
                
                if transformed_image != None:
                    images.append(transformed_image['image'])
                    groundtruths.append(transformed_image['groundtruth'])
        
        return images, groundtruths

    def _get_transforms(self):
        transforms_list = []
        
        padding = (self.large_patch_size - self.patch_size)//2
        
        im_size1 = (self.image_initial_size+ 2*padding)
        im_size2 = self.image_initial_size
        
        rot_padding = math.ceil(math.ceil((im_size1 * math.sqrt(2)) - im_size2) / 2)
    
        transform0 = transformations.Pad(padding, padding_mode="symmetric")
        
        transforms_list.append(transform0)
    
        transform1 = transforms.Compose(
            [transform0, transformations.RandomHorizontalFlip(p=1)]
        )
        
        transforms_list.append(transform1)
    
        transform2 = transforms.Compose(
            [transform0, transformations.RandomVerticalFlip(p=1)]
        )
        
        transforms_list.append(transform2)
    
        transform3 = transforms.Compose(
            [transformations.Pad(rot_padding, padding_mode="symmetric"), 
             transformations.RandomRotation(degrees=90), 
             transformations.CenterCrop(self.image_initial_size+2*padding)
            ]
        )

        transforms_list.append(transform3)
    
        transform4 = transforms.Compose(
            [transform0, transformations.ColorJitter()]
        )
        
        transforms_list.append(transform4)
        
        return transforms_list
        

class RoadsDatasetTest(Dataset):
    """Road segmentation dataset for test time"""

    def __init__(self,patch_size, large_patch_size, number_patch_per_image, image_initial_size,root_dir):
        self.root_dir = Path(root_dir)
        self.img_names = [str(x) for x in self.root_dir.glob("**/*.png") if x.is_file()]
        
        self.patch_size = patch_size
        self.large_patch_size = large_patch_size
        self.number_patch_per_image = number_patch_per_image
        self.image_initial_size = image_initial_size
        
        self.transforms = None
        self.images = self._extract_images()
        
    def __len__(self):
        return self.number_patch_per_image * len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
       
        l_p_s = self.large_patch_size
        n_p_p_i = self.number_patch_per_image
        n_s_p_p_i = self.image_initial_size // self.patch_size
        p_s = self.patch_size
        image_index = idx//n_p_p_i
        patch_index = idx%n_p_p_i
        
        padding = (l_p_s - p_s) // 2
        
        y = ((patch_index % n_s_p_p_i) * p_s)+padding
        x = ((patch_index// n_s_p_p_i) * p_s)+padding
        
        image = self.images[image_index]
        
        small_image = F.crop(image, x-padding, y-padding, l_p_s,l_p_s)
        
        x = (x-padding)//p_s
        y = (y-padding)//p_s
        
        sample = small_image
       
        transformation = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        
        sample = transformation(sample)
            
        sample = {"id": image_index, "x" : x, "y" : y, "image": sample}

        return sample
    
    def _extract_images(self):
        images = []
        
        padding = (self.large_patch_size - self.patch_size)//2
        
        for i in range(len(self.img_names)):
            name = self.img_names[i]
            image = Image.open(name)
            transformed_image = transforms.Pad(padding, padding_mode="symmetric")(image)
            images.append(transformed_image)
                
        return images