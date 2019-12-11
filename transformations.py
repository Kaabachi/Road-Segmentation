import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision import functional as F
import random


class ToTensor(transforms.ToTensor):
    def __call__(self, sample):
        image = sample["image"]
        groundtruth = sample["groundtruth"]
        return {"image": F.to_tensor(image), "groundtruth": F.to_tensor(groundtruth)}


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        """
        Args:
            sample (dict of PIL Images): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return {
                "image": F.hflip(sample["image"]),
                "groundtruth": F.hflip(sample["groundtruth"]),
            }
        return sample


class RandomVerticalFlip(transforms.RandomVerticalFlip):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __call__(sefl, sample):
        if random.random() < self.p:
            return {
                "image": F.vflip(sample["image"]),
                "groundtruth": F.vflip(sample["groundtruth"]),
            }
        return sample


class RandomRotation(transforms.RandomRotation):
    """Rotate the image by angle.

        Args:
            degrees (sequence or float or int): Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees).
            resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
                An optional resampling filter. See `filters`_ for more information.
                If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
            expand (bool, optional): Optional expansion flag.
                If true, expands the output to make it large enough to hold the entire rotated image.
                If false or omitted, make the output image the same size as the input image.
                Note that the expand flag assumes rotation around the center and no translation.
            center (2-tuple, optional): Optional center of rotation.
                Origin is the upper left corner.
                Default is the center of the image.
            fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
                If int, it is used for all channels respectively.

        .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

        """

    def __call__(self, sample):
        angle = self.get_params(self.degrees)
        image = sample["image"]
        groundtruth = sample["groundtruth"]
        return {
            "image": F.rotate(
                image, angle, self.resample, self.expand, self.center, self.fill
            ),
            "groundtruth": F.rotate(
                groundtruth, angle, self.resample, self.expand, self.center, self.fill
            ),
        }


class ColorJitter(transforms.ColorJitter):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __call__(self, sample):
        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        # Only transorm the image, the mask shouldn't change
        return {
            "image": transform(sample["image"]),
            "groundtruth": sample["groundtruth"],
        }


class Normalize(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return {
            "image": F.normalize(sample["image"], self.mean, self.std, self.inplace),
            "groundtruth": sample["groundtruth"],
        }


class Resize(transforms.Resize):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __call__(self, sample):
        resized_image = F.resize(sample["image"], self.size, self.interpolation)
        resized_groundtruth = F.resize(
            sample["groundtruth"], self.size, self.interpolation
        )
        return {"image": resized_image, "groundtruth": resized_groundtruth}
