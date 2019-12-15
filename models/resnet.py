import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

PRETRAINED = False
NUM_CLASSES = 1


class ResNet(nn.Module):
    def __init__(self, resnet_version=50):
        super(ResNet, self).__init__()
        if resnet_version == 50:
            self.fcn_resnet = models.segmentation.fcn_resnet50(
                pretrained=PRETRAINED, num_classes=NUM_CLASSES
            )
        elif resnet_version == 101:
            self.fcn_resnet = models.segmentation.fcn_resnet101(
                pretrained=PRETRAINED, num_classes=NUM_CLASSES
            )
        else:
            raise Exception("resnet must be 50 or 101")
        self.model_name = f"fcn_resnet_{resnet_version}"

    def forward(self, x):
        up_image = F.interpolate(x, size=224)
        output = self.fcn_resnet(up_image)['out']
        down_image = F.interpolate(output, size=96)
        return down_image.clamp(0.0, 1.0)

