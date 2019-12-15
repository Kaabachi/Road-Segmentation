import torch.nn as nn
import torch

from collections import OrderedDict

WITH_BIASES = True

def block(in_channels, out_channels, name):
    return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=WITH_BIASES,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                            bias=WITH_BIASES,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=out_channels)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_filters=32):
        super(UNet, self).__init__()
        self.model_name = "unet"
        filters = init_filters
        self.encoder1 = block(in_channels, filters, name="encoder1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = block(filters, filters * 2, name="encoder2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = block(filters * 2, filters * 4, name="encoder3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = block(filters * 4, filters * 8, name="encoder4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottom = block(filters * 8, filters * 16, name="bottom")

        self.upconv4 = nn.ConvTranspose2d(
            filters * 16, filters * 8, kernel_size=2, stride=2
        )
        self.decoder4 = block((filters * 8) * 2, filters * 8, name="decoder4")
        
        self.upconv3 = nn.ConvTranspose2d(
            filters * 8, filters * 4, kernel_size=2, stride=2
        )
        self.decoder3 = block((filters * 4) * 2, filters * 4, name="decoder3")
        
        self.upconv2 = nn.ConvTranspose2d(
            filters * 4, filters * 2, kernel_size=2, stride=2
        )
        self.decoder2 = block((filters * 2) * 2, filters * 2, name="decoder2")
        
        self.upconv1 = nn.ConvTranspose2d(
            filters * 2, filters, kernel_size=2, stride=2
        )
        self.decoder1 = block(filters * 2, filters, name="decoder1")

        self.finalconv = nn.Conv2d(
            in_channels=filters, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottom = self.bottom(self.pool4(enc4))

        dec4 = self.upconv4(bottom)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.finalconv(dec1))