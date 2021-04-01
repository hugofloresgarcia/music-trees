from functools import reduce
from operator import __add__

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 3),
                 stride=1, padding='same'):
        super().__init__()
        assert isinstance(kernel_size, tuple)

        if padding == 'same' and stride == 1:
            padding = reduce(__add__,
                             [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        else:
            raise ValueError(
                ' implemented anything other than same padding and stride 1')

        self.pad = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(in_channels=in_channels,  out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels, momentum=0.9)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.pad(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.b_norm = nn.BatchNorm2d(1)
        self.conv1 = ConvBlock(
            in_channels=1, out_channels=128, kernel_size=(3, 3))
        self.conv2 = ConvBlock(
            in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv3 = ConvBlock(
            in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv4 = ConvBlock(
            in_channels=128, out_channels=128, kernel_size=(3, 3))

    def forward(self, x):
        # input should be shape (batch, channels, frequency, time)
        x = self.b_norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # pool over time dimension
        d_time = x.shape[-1]
        x = F.max_pool2d(x, kernel_size=(1, d_time))

        # reshape to (batch, feature)
        d_batch = x.shape[0]
        x = x.view(d_batch, -1)

        return x
