import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision.transforms import functional as F

import transformations
from helpers import natural_keys, pad_image, to_int

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


VALIDATION_ID_THRESHOLD = 90


class RoadsDatasetTrain(Dataset):
    """Road segmentation datset"""

    def __init__(
        self,
        patch_size,
        large_patch_size,
        number_patch_per_image,
        image_initial_size,
        root_dir,
    ):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / "images"
        self.gt_dir = self.root_dir / "groundtruth"
        self.img_names = [x.name for x in self.img_dir.glob("**/*.png") if x.is_file()]


        self.img_names = [x.name for x in self.img_dir.glob("**/*.png") if x.is_file()]
        # keep only image with id >= 90
        self.img_names = [
            x
            for x in self.img_names
            if int(x.split("_")[1].split(".")[0]) < VALIDATION_ID_THRESHOLD
        ]


        self.large_patch_size = large_patch_size
        self.number_patch_per_image = number_patch_per_image
        self.image_initial_size = image_initial_size
        self.patch_size = patch_size

        # self.img_transform = transforms.Compose(
        # [
        # transformations.ToPILImage(),
        ## TODO: Put right parameters for pad to get to 480x480
        # transformations.Pad((self.large_patch_size - self.patch_size)//2)
        # ]
        # )
        self.transforms = self._get_transforms()

    def __len__(self):
        #number_patch_per_image is the number of 96x96 patches that can be extracted from a 400x400 image
        #Those 96x96 patches overlap in plenty of pixels, apart from the 16x16 center patches
        return self.number_patch_per_image * len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        l_p_s = self.large_patch_size
        n_p_p_i = self.number_patch_per_image
        n_s_p_p_i = self.image_initial_size // self.patch_size
        p_s = self.patch_size
        l_p_s = self.large_patch_size
       
    
        #image_index is in the range of (0, nb_images * nb_transforms-1)
        image_index = idx // n_p_p_i
 
        #patch_index is in the range of (0, number_patch_per_image-1)
        patch_index = idx % n_p_p_i

        padding = (l_p_s - p_s) // 2

        #computing x,y coordinates of the top-left corner of the patch
        y = ((patch_index % n_s_p_p_i) * p_s) + padding
        x = ((patch_index // n_s_p_p_i) * p_s) + padding

        image_sample = self._get_image_and_gt(image_index)
        image = image_sample["image"]
        groundtruth = image_sample["groundtruth"]

        #Cropping only the needed patches
        small_image = F.crop(image, x - padding, y - padding, l_p_s, l_p_s)
        small_groundtruth = F.crop(groundtruth, x - padding, y - padding, l_p_s, l_p_s)

        sample = {"image": small_image, "groundtruth": small_groundtruth}

        transformation = transformations.ToTensor()
        sample = transformation(sample)

        return sample

    def _get_image_and_gt(self, image_index):
        """
        Will get the image and grond truth corresponding to index and will transform them according to self.img_transform
        """
        image_name = self.img_names[image_index]

        image = Image.open(self.img_dir / image_name)
        groundtruth = Image.open(self.gt_dir / image_name)
        image_sample = {"image": image, "groundtruth": groundtruth}

        if self.transforms is not None:
            image_sample = self.transforms(image_sample)
        return image_sample

    def _get_transforms(self):

        padding = (self.large_patch_size - self.patch_size) // 2

        im_size1 = self.image_initial_size + 2 * padding
        im_size2 = self.image_initial_size

        rot_padding = math.ceil(math.ceil((im_size1 * math.sqrt(2)) - im_size2) / 2)
        transforms_list = [
            
            #Pads the image with the given padding and fill the surplus of pixels by mirroring 
            transformations.Pad(padding, padding_mode="symmetric"),
            
            #Apply Horizontal flip
            transformations.RandomHorizontalFlip(p=1),
            
            #Apply Vertical flip
            transformations.RandomVerticalFlip(p=1),
            
            #Pads the image with the given padding and fill the surplus of pixels by mirroring 
            transformations.Pad(rot_padding, padding_mode="symmetric"),
            
            #Apply rotation to image
            transformations.RandomRotation(degrees=90),
            
            #Crop image at the center
            transformations.CenterCrop(self.image_initial_size + 2 * padding),
        
            #Randomly jitters the brightness, contrast, saturation and hue of the image
            transformations.ColorJitter(),
            
            #Changes image to Tensor
            transformations.ToTensor(),
            # TODO: Right params for normalize, can be the mean and std of imageNet

            transformations.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transformations.ToPILImage(),
        ]

        return transforms.Compose(transforms_list)


class RoadsDatasetTest(Dataset):
    """Road segmentation dataset for test time"""

    def __init__(
        self,
        patch_size,
        large_patch_size,
        number_patch_per_image,
        image_initial_size,
        root_dir,
    ):
        self.root_dir = Path(root_dir)
        self.img_names = [str(x) for x in self.root_dir.glob("**/*.png") if x.is_file()]


        
        self.patch_size = patch_size
        self.large_patch_size = large_patch_size
        self.number_patch_per_image = number_patch_per_image
        self.image_initial_size = image_initial_size

    def __len__(self):
        #number_patch_per_image is the number of 96x96 patches that can be extracted from a 608x608 image
        #Those 96x96 patches overlap in plenty of pixels, apart from the 16x16 center patches
        return self.number_patch_per_image * len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        l_p_s = self.large_patch_size
        n_p_p_i = self.number_patch_per_image
        n_s_p_p_i = self.image_initial_size // self.patch_size
        p_s = self.patch_size
        image_index = idx // n_p_p_i
        patch_index = idx % n_p_p_i

        padding = (l_p_s - p_s) // 2

        y = ((patch_index % n_s_p_p_i) * p_s) + padding
        x = ((patch_index // n_s_p_p_i) * p_s) + padding

        image = self._extract_image(image_index)

        small_image = F.crop(image, x - padding, y - padding, l_p_s, l_p_s)

        x = (x - padding) // p_s
        y = (y - padding) // p_s

        sample = small_image

        transformation = transforms.Compose([transforms.ToTensor()])

        sample = transformation(sample)

        sample = {"id": image_index, "x": x, "y": y, "image": sample}

        return sample

    def _extract_image(self, image_index):
        padding = (self.large_patch_size - self.patch_size) // 2
        image_name = self.img_names[image_index]
        image = Image.open(image_name)
        transformation = transforms.Compose(
            [
            # ImageNet normalization for now
            transforms.Pad(padding, padding_mode="symmetric"),
            transforms.ToTensor(),
            # TODO: Right params for normalize, can be the mean and std of imageNet
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.ToPILImage(),

            ]
        )
        return transformation(image)
    


class RoadsDatasetValidation(Dataset):
    """Road segmentation dataset for test time"""

    def __init__(
        self,
        patch_size,
        large_patch_size,
        number_patch_per_image,
        image_initial_size,
        root_dir,
    ):
        self.root_dir = Path(root_dir)
        self.img_names = [str(x) for x in self.root_dir.glob("**/*.png") if x.is_file()]
        # keep only image with id >= 90
        self.img_names = [
            x
            for x in self.img_names
            if int(x.split("_")[1].split(".")[0]) >= VALIDATION_ID_THRESHOLD
        ]

        self.patch_size = patch_size
        self.large_patch_size = large_patch_size
        self.number_patch_per_image = number_patch_per_image
        self.image_initial_size = image_initial_size

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
        image_index = idx // n_p_p_i
        patch_index = idx % n_p_p_i

        padding = (l_p_s - p_s) // 2

        y = ((patch_index % n_s_p_p_i) * p_s) + padding
        x = ((patch_index // n_s_p_p_i) * p_s) + padding

        image_sample = self._extract_image(image_index)
        image = image_sample["image"]
        groundtruth = image_sample["groundtruth"]

        small_image = F.crop(image, x - padding, y - padding, l_p_s, l_p_s)
        small_groundtruth = F.crop(groundtruth, x - padding, y - padding, l_p_s, l_p_s)

        x = (x - padding) // p_s
        y = (y - padding) // p_s

        transformation = transforms.ToTensor()

        image = transformation(image)
        groundtruth = transformation(groundtruth)

        sample = {
            "id": image_index,
            "x": x,
            "y": y,
            "image": small_image,
            "groundtruth": groundtruth,
        }

        return sample

    def _extract_image(self, image_index):
        padding = (self.large_patch_size - self.patch_size) // 2
        image_name = self.img_names[image_index]
        image = Image.open(self.img_dir / image_name)
        groundtruth = Image.open(self.gt_dir / image_name)
        sample = {"image": image, "groundtruth": groundtruth}
        transformation = transforms.Compose(
            [
                # ImageNet normalization for now
                transformations.ToTensor(),
                transformations.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                transformations.ToPILImage(),
                transforms.Pad(padding, padding_mode="symmetric"),

            ]
        )
        return transformation(sample)
