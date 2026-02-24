import torch
import torch.nn as nn

class SingleStageDecoder(nn.Module):
    def __init__(self, num_classes = 2, in_channels = 512):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, num_classes, kernel_size = 64, stride = 32, padding = 16)

    def forward(self, x2, x3, x4):
        return self.up_conv(x4)
    
class ProgressiveDecoder(nn.Module):
    def __init__(self, num_classes = 2):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1)
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size = 3, padding = 1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace = True))

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size = 3, padding = 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace = True))

        self.up3 = nn.ConvTranspose2d(128, num_classes, kernel_size = 16, stride = 8, padding = 4)

    def forward(self, x2, x3, x4):
        d1 = self.up1(x4)
        d1 = torch.cat([d1, x3], dim = 1)
        d1 = self.conv1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim = 1)
        d2 = self.conv2(d2)

        d3 = self.up3(d2)
        return d3
    