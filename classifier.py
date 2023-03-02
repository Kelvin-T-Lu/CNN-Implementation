import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Convolution_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.batch_norm(self.conv(x))


class Residual_Block(nn.Module):
    """ Residual blocks for bottlenecks. """
    def __init__(self, in_channels, out_channels, first_block=False):
        super().__init__()
        
        res_channels = in_channels // 4
        stride = 1

        self.projection = in_channels!=out_channels
        if self.projection:
            self.p = Convolution_Block(in_channels, out_channels, 1, 2, 0)
            stride = 2
            res_channels = in_channels // 2

        if first_block:
            self.p = Convolution_Block(in_channels, out_channels, 1, 1, 0)
            stride = 1
            res_channels = in_channels

        self.conv1 = Convolution_Block(in_channels, res_channels, 1, 1, 0) 
        self.conv2 = Convolution_Block(res_channels, res_channels, 3, stride, 1)
        self.conv3 = Convolution_Block(res_channels, out_channels, 1, 1, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.relu(self.conv1(x))
        f = self.relu(self.conv2(f))
        f = self.conv3(f)

        if self.projection:
            x = self.p(x)

        g = self.relu(torch.add(f, x))
        return g

class Classifier(nn.Module):
    """ Resnet Implementation"""
    def __init__(self):
        
        super().__init__()

        # Simulate Res50
        block_num = [3, 4, 6, 3]

        out_res_channels = [256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([Residual_Block(64, 256, True)])

        for i in range(len(out_res_channels)):
            if i > 0:
                self.blocks.append(Residual_Block(out_res_channels[i-1], out_res_channels[i]))
            for _ in range(block_num[i]-1):
                self.blocks.append(Residual_Block(out_res_channels[i], out_res_channels[i]))
        
        self.conv1 = Convolution_Block(3, 64, 7, 2, 3)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mean_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, NUM_CLASSES)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)

        x = self.mean_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x