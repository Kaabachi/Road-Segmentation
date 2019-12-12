import torch.nn as nn
from torchvision import models



#C_DIM, X_DIM, Y_DIM = 3, 96, 96

PRETRAINED = False
NUM_CLASSES = 2


fcn_resnet101 = models.segmentation.fcn_resnet101(pretrained=PRETRAINED, num_classes=NUM_CLASSES)
fcn_resnet50 = models.segmentation.fcn_resnet50(pretrained=PRETRAINED, num_classes=NUM_CLASSES)