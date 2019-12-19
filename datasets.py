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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALIDATION_ID_THRESHOLD = 90


class RoadsDatasetTrain(Dataset):
    """Road segmentation datset"""

    def __init__(self,patch_size, large_patch_size, number_patch_per_image, image_initial_size, root_dir):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.gt_dir = self.root_dir / "groundtruth"
        
        self.img_names = [x.name for x in self.img_dir.glob("**/*.png") if x.is_file()]


        self.large_patch_size = large_patch_size
        self.number_patch_per_image = number_patch_per_image
        self.image_initial_size = image_initial_size
        self.patch_size = patch_size
        
        #Get the list of transformations to be used for Data Augmentation on the whole image
        self.whole_image_transforms = self._get_whole_image_transforms()
        
        #Get the list of transformations to be used on individual 96x96 patches when sampled
        self.patch_transforms = self._get_patch_transforms()
        
        #Load all 400x400 images and groundtruths to memory and applying whole image transformations on them
        #Gives a length of nb_images * nb_transforms
        self.images, self.groundtruths = self._extract_images()

    def __len__(self):
        #number_patch_per_image is the number of 96x96 patches that can be extracted from a 400x400 image
        #Those 96x96 patches overlap in plenty of pixels, apart from the 16x16 center patches
        return self.number_patch_per_image * len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        l_p_s = self.large_patch_size
        n_p_p_i = self.number_patch_per_image
        n_s_p_p_i = self.image_initial_size // self.patch_size
        p_s = self.patch_size
        l_p_s = self.large_patch_size
        
        #image_index is in the range of (0, nb_images * nb_transforms-1)
        image_index = idx//n_p_p_i
        
        #patch_index is in the range of (0, number_patch_per_image-1)
        patch_index = idx%n_p_p_i
        
        padding = (l_p_s - p_s) // 2
        
        #computing x,y coordinates of the top-left corner of the patch
        y = ((patch_index % n_s_p_p_i) * p_s)+padding
        x = ((patch_index // n_s_p_p_i) * p_s)+padding

        image = self.images[image_index]
        groundtruth = self.groundtruths[image_index]
        
        #Cropping only the needed patches
        small_image = F.crop(image,x-padding,y-padding,l_p_s,l_p_s)
        small_groundtruth = F.crop(groundtruth,x-padding,y-padding,l_p_s,l_p_s)
        
        sample = {"image": small_image, "groundtruth": small_groundtruth}
        
        #Applying patch transforms before returning the sample
        transformation = self.patch_transforms
        
        sample = transformation(sample)

        return sample
    
    #Extract images and their transformed versions from disk and loading them to memory without dividing them to patches
    def _extract_images(self):
        images = []
        groundtruths = []
  
        transforms = self.whole_image_transforms
        
        for i in range(len(self.img_names)):
            name = self.img_names[i]
            image = Image.open(self.img_dir / name)
            groundtruth = Image.open(self.gt_dir / name)
            
            sample = {'image':image, 'groundtruth':groundtruth}
            
            for j in range(len(transforms)):
                transformed_image = transforms[j](sample)
                images.append(transformed_image['image'])
                groundtruths.append(transformed_image['groundtruth'])
        
        return images, groundtruths

    def _get_patch_transforms(self):
        transform = transforms.Compose(
            [transformations.RandomVerticalFlip(),
             transformations.RandomHorizontalFlip(),
             transformations.RandomRotation(degrees=90),
             transformations.CenterCrop(self.large_patch_size),
             transformations.ToTensor(),
             transformations.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]
        )
        
        return transform
    
    def _get_whole_image_transforms(self):
        transforms_list = []
        
        padding = (self.large_patch_size - self.patch_size)//2
        
        im_size1 = (self.image_initial_size+ 2*padding)
        im_size2 = self.image_initial_size
        
        rot_padding = math.ceil(math.ceil((im_size1 * math.sqrt(2)) - im_size2) / 2)
        
        #Pads the image with the given padding and fill the surplus of pixels by mirroring 
        transform0 = transformations.Pad(padding, padding_mode="symmetric")
        
        transforms_list.append(transform0)

        
        return transforms_list
        

#Pretty much same structure as the RoadsDatasetTrain
class RoadsDatasetTest(Dataset):
    """Road segmentation dataset for test time"""

    def __init__(self,patch_size, large_patch_size, number_patch_per_image, image_initial_size,root_dir):
        self.root_dir = Path(root_dir)
        self.img_names = [str(x) for x in self.root_dir.glob("**/*.png") if x.is_file()]
        # Sort images to in a human readable way
        self.img_names.sort(key=natural_keys)

        
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
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
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