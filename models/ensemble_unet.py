import torch.nn as nn
import torch
from models.unet import UNet
from torchvision.transforms import functional as F
from torchvision import transforms

class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.model_name = "ensemble-unet"
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

#         print(infos_flip.size())
#         print(infos_angle.size())
        
        for i in range(infos_angle.size()[0]):
            for j in range(infos_angle.size()[1]):
                if infos_angle[i][j] != 0 :
                    outputs[j][i] = transforms.ToTensor()(
                        F.rotate(
                            transforms.ToPILImage()(outputs[j][i]),
                            -infos_angle[i][j],
                            resample=False,
                            expand=False,
                            center=None
                        )
                    )

                if infos_flip[i][j][0]:
                    outputs[j][i] = transforms.ToTensor()(
                        F.hflip(
                            transforms.ToPILImage()(outputs[j][i])
                            )
                    )

                if infos_flip[i][j][1]:
                    outputs[j][i] = transforms.ToTensor()(
                        F.vflip(
                            transforms.ToPILImage()(outputs[j][i])
                        )
                    )
                
#         return torch.mean(outputs, dim=1, keepdim=True)
        
        stacked = torch.cat(outputs, dim=1)
        
        final = self.finalconv(stacked)
        
        return torch.sigmoid(final)