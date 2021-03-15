"""model.py - model definition"""

from functools import reduce
from operator import __add__
import argparse
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 3), 
                 stride=1, padding = 'same'):
        super().__init__()
        assert isinstance(kernel_size, tuple)

        if padding == 'same' and stride == 1:
            padding = reduce(__add__,
                                [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        else:
            raise ValueError(' implemented anything other than same padding and stride 1')

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

        self.b_norm = nn.BatchNorm2d(1)
        self.conv1 = ConvBlock(in_channels=1, out_channels=128, kernel_size=(3, 3))
        self.conv2 = ConvBlock(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv3 = ConvBlock(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.conv4 = ConvBlock(in_channels=128, out_channels=128, kernel_size=(3, 3))

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


class Model(pl.LightningModule):
    """PyTorch Lightning model definition"""

    def __init__(self, parents: list):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = Backbone()

        # given a parent node in the hierarchy, there will be 
        # one linear layer that maps from the backbone embedding
        # to the parent node
        root_d = None
        parent_d = root_d // len(parents)
        self.parents = nn.ModuleList([nn.Linear(root_d, parent_d) for name in parents])
        
        # this fully connected layer will serve as our classifier for the 
        # parent level
        self.fc = nn.Linear(parent_d, len(parents))

    def _forward(self, x):
        """forward pass through the backbone model, as well as the 
        coarse grained classifier. Returns the output, weighed embedding, 
        as well as the probits
        """
        # input should be shape (b, c, f, t)
        x = self.backbone(x)

        # now, forward pass through each parent layer
        # each tensor in this list should be shape (b, e)
        embeddings = [layer(x) for  layer in self.parents.items()]

        # concatenate embeddings to make a single feature vector
        x = torch.cat(embeddings, dim=-1)

        # classify vector
        probits = self.fc(x) 

        # use these probits to weigh the embeddings 
        x = torch.cat([emb * probit for emb, probit in zip(embeddings, probits)], dim=-1)

        # return the vector and the probits
        return x, probits

    def forward(self, support, query):
        raise NotImplementedError

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model hyperparameters as argparse arguments"""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # TODO - add hyperparameters as command-line args using
        #        parser.add_argument()
        return parser

    def training_step(self, batch, index):
        """Performs one step of training"""
        # TODO - implement training step
        raise NotImplementedError

    def validation_step(self, batch, index):
        """Performs one step of validation"""
        # TODO - implement validation step
        raise NotImplementedError

    def test_step(self, batch, index):
        """Performs one step of testing"""
        # OPTIONAL - only implement if you have meaningful objective metrics
        raise NotImplementedError

    def configure_optimizer(self):
        """Configure optimizer for training"""
        return torch.optim.Adam(self.parameters())
