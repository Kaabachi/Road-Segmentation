import torch.nn as nn
import torch
from models.unet import UNet
from torchvision.transforms import functional as F
from torchvision import transforms

class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.model0 = UNet()
        self.model1 = UNet()
        self.model2 = UNet()
        self.model3 = UNet()
        self.model4 = UNet()
        
        self.finalconv = nn.Conv2d(
            in_channels=5, out_channels=1, kernel_size=1
        )
        
    def forward(self, x, infos_angle, infos_flip):
        x_models = []
        
        x_models.append(x[:,:3])
        x_models.append(x[:,3:6])
        x_models.append(x[:,6:9])
        x_models.append(x[:,9:12])
        x_models.append(x[:,12:15])
        
        outputs = []
        
        outputs.append(self.model0(x_models[0]))
        outputs.append(self.model1(x_models[1]))
        outputs.append(self.model2(x_models[2]))
        outputs.append(self.model3(x_models[3]))
        outputs.append(self.model4(x_models[4]))
        
        for k in range(len(infos_angle)):
            for i in range(len(infos_angle[k])):
                
                if infos_angle[k][i] != 0 :
                    outputs[k][i] = transforms.ToTensor()(
                        F.rotate(
                            transforms.ToPILImage()(outputs[k][i]),
                            -infos_angle[k][i],
                            resample=False,
                            expand=False,
                            center=None
                        )
                    )

                if infos_flip[k][i][0]:
                    outputs[k][i] = transforms.ToTensor()(
                        F.hflip(
                            transforms.ToPILImage()(outputs[k][i])
                            )
                    )

                if infos_flip[k][i][1]:
                    outputs[k][i] = transforms.ToTensor()(
                        F.vflip(
                            transforms.ToPILImage()(outputs[k][i])
                        )
                    )
                
#         return torch.mean(outputs, dim=1, keepdim=True)
        
        stacked = torch.cat(outputs, dim=1)
        
        final = self.finalconv(stacked)
        
        return torch.sigmoid(final)