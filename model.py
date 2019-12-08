import torch.nn as nn



C_DIM, X_DIM, Y_DIM = 3, 96, 96


#temp name
class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3),
            nn.Relu(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.Relu(),
            nn.Conv2d(128, 128, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.Relu(),
            nn.Conv2d(256, 256, 3),
            nn.Relu(),
            nn.MaxPool2d(2)
        )