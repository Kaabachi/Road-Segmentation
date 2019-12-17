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
        # Sort images to in a human readable way
        self.img_names.sort(key=natural_keys)

        self.large_patch_size = large_patch_size
        self.number_patch_per_image = number_patch_per_image
        self.image_initial_size = image_initial_size
        self.patch_size = patch_size

        self.img_transform = transforms.Compose(
            [
                transformations.ToPILImage(),
                # TODO: Put right parameters for pad to get to 480x480
                transformations.Pad(),
            ]
        )
        self.patch_transforms = self._get_transforms()
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

        image_index = idx // n_p_p_i
        patch_index = idx % n_p_p_i

        padding = (l_p_s - p_s) // 2

        y = ((patch_index % n_s_p_p_i) * p_s) + padding
        x = ((patch_index // n_s_p_p_i) * p_s) + padding

        image_sample = self._get_image_and_gt(image_index)
        image = image_sample["image"]
        groundtruth = image_sample["groundtruth"]

        small_image = F.crop(image, x - padding, y - padding, l_p_s, l_p_s)
        small_groundtruth = F.crop(groundtruth, x - padding, y - padding, l_p_s, l_p_s)

        sample = {"image": small_image, "groundtruth": small_groundtruth}

        transformation = self.patch_transforms
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

        if self.img_transforms is not None:
            image_sample = self.img_transforms(image_sample)
        return image_sample


    def _get_transforms(self):

        padding = (self.large_patch_size - self.patch_size) // 2

        im_size1 = self.image_initial_size + 2 * padding
        im_size2 = self.image_initial_size

        rot_padding = math.ceil(math.ceil((im_size1 * math.sqrt(2)) - im_size2) / 2)
        transforms_list = [
            transformations.Pad(padding, padding_mode="symmetric"),
            transformations.RandomHorizontalFlip(p=1),
            transformations.RandomVerticalFlip(p=1),
            transformations.Pad(rot_padding, padding_mode="symmetric"),
            transformations.RandomRotation(degrees=90),
            transformations.CenterCrop(self.image_initial_size + 2 * padding),
            transformations.ColorJitter(),
            transformations.ToTensor(),
            #TODO: Right params for normalize, can be the mean and std of imageNet
            transformations.Normalize()
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
        image_index = idx // n_p_p_i
        patch_index = idx % n_p_p_i

        padding = (l_p_s - p_s) // 2

        y = ((patch_index % n_s_p_p_i) * p_s) + padding
        x = ((patch_index // n_s_p_p_i) * p_s) + padding

        image = self.images[image_index]

        small_image = F.crop(image, x - padding, y - padding, l_p_s, l_p_s)

        x = (x - padding) // p_s
        y = (y - padding) // p_s

        sample = small_image

        transformation = transforms.Compose([transforms.ToTensor()])

        sample = transformation(sample)

        sample = {"id": image_index, "x": x, "y": y, "image": sample}

        return sample

    def _extract_images(self):
        images = []

        padding = (self.large_patch_size - self.patch_size) // 2

        for i in range(len(self.img_names)):
            name = self.img_names[i]
            image = Image.open(name)
            transformed_image = transforms.Pad(padding, padding_mode="symmetric")(image)
            images.append(transformed_image)

        return images