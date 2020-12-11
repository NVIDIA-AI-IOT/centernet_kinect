import os
import sys
import torch
import torchvision
import torch.nn as nn

# Adding Project Path
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(PROJ_PATH)

# Importing Project Libraries
import pipeline.constants as const

class CRBUp(nn.Module):
    """
    Convolution Residual Block Upsampling Class
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(CRBUp, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layers(x)


class Resnet18FeatureExtractor(nn.Module):

    def __init__(self, num_classes=const.NUM_CLASSES, pretrained=True):
        super(Resnet18FeatureExtractor, self).__init__()
        self.num_classes = num_classes
        self.out_channels = 4 + num_classes
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        
        self.up_sample1 = CRBUp(512, 256)
        self.up_sample2 = CRBUp(512, 128)
        self.up_sample3 = CRBUp(256, 64)
        self.up_sample4 = CRBUp(128, self.out_channels)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        # Upsampling 
        out = self.up_sample1(x4)
        out = self.up_sample2(torch.cat([x3, out], 1))
        out = self.up_sample3(torch.cat([x2, out], 1))
        out = self.up_sample4(torch.cat([x1, out], 1))

        out[:,0:self.num_classes] = self.sigmoid(out[:,0:self.num_classes])
        
        return out