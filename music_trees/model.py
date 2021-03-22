"""model.py - model definition"""

from functools import reduce
from operator import __add__
import argparse
import logging
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_detach(dict_of_tensors):
    for k, v in dict_of_tensors.items():
        if isinstance(v, torch.Tensor):
            dict_of_tensors[k] = v.detach()
    return dict_of_tensors

def batch_cpu(dict_of_tensors):
    for k, v in dict_of_tensors.items():
        if isinstance(v, torch.Tensor):
            dict_of_tensors[k] = v.cpu()
    return dict_of_tensors

batch_detach_cpu = lambda x: batch_cpu(batch_detach(x))

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
        super().__init__()
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

class ProtoNet(pl.LightningModule):
    """PyTorch Lightning model definition"""

    def __init__(self, parents: list):
        super().__init__()
        self.save_hyperparameters()

        self.alpha = torch.tensor(0.2)
        self.backbone = Backbone()

        # do a dummy forward pass through the 
        # embedding model to get the output shape
        dummy_input = torch.randn((1, 1, 128, 199))
        dummy_out = self.backbone(dummy_input)
        root_d = dummy_out.shape[-1]

        # given a parent node in the hierarchy, there will be 
        # one linear layer that maps from the backbone embedding
        # to a smaller embedding
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
        """ 
        support should be a torch Tensor of shape (b(atch), c(lass), k, 1, f(requency), t(ime))
        query should be a torch Tensor of shape (b, q, 1, f(requency), t(ime))

        output will be a dict with
        
        dists: euclidean distances formed from comparing query vectors with support prototypes, with shape (b, q, c)
        query_probits: log-probabilities for query examples, shape (b, q, p(arents)
        support_probits: log-probabilities for each example in the (b(atch), c(lass), k, p(arents))
        """
        assert support.ndim == 6
        assert query.ndim == 5

        # get support embeddings and parent probits
        b_dim, c_dim, k_dim, _, f_dim, t_dim = support.shape()
        support = support.view(b_dim*c_dim*k_dim, 1, f_dim, t_dim)
        support, support_probits = self._forward(support)
        
        support = support.view(b_dim, c_dim, k_dim, -1) # expand back
        support_probits = support_probits.view(b_dim, c_dim, k_dim, -1) # expand probits as well

        # take mean over k_dim to get the prototypes
        prototypes = support.mean(dim=2, keepdim=False)

        # get query embeddings
        b_dim, q_dim, _, f_dim, t_dim = query.shape()
        query = query.view(b_dim * q_dim, 1, f_dim, t_dim)
        query, query_probits = self._forward(query)

        query = query.view(b_dim, q_dim, -1)
        query_probits = query_probits.view(b_dim, q_dim, -1)

        # by here, query should be shape (b, q, n)
        # and prototypes should be shape (b, c, n)

        # compute euclidean distances
        # output shoudl be shape (b, q, c)
        # so that the row vectors are the logits for the classes 
        # for each query, in the batch
        dists = torch.cdist(query, prototypes, p=2)

        return {'distances': dists, 
                'query_probits': query_probits, 
                'support_probits': support_probits}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model hyperparameters as argparse arguments"""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # TODO - add hyperparameters as command-line args using
        #        parser.add_argument()
        return parser

    @staticmethod
    def criterion(ypred, ytrue):
        # in this case, our probits will be shape (batch, query, n)
        # and our targets will be (batch, query)
        assert ypred.ndim == 3
        assert ytrue.ndim == 2

        b, q, _ = ypred.shape
        ypred = ypred.view(b * q, -1)
        ytrue = ytrue.view(-1)

        return F.cross_entropy(ypred, ytrue)

    def _squash_parent_targets(batch):
        support_targets = batch['support_targets']
        support_targets = support_targets.view(-1)

        query_targets = batch['query_targets']
        query_targets = query_targets.view(-1)

        return torch.cat([support_targets, query_targets], dim=-1)

    def _squash_parent_probits(output):
        support_probits = output['support_probits']
        support_probits = support_probits.view(-1, support_probits.shape[-1])

        query_probits = output['query_probits']
        query_probits = query_probits.view(-1, support_probits.shape[-1])
        
        return torch.cat([support_probits, query_probits], dim=-1)

    def _main_step(self, batch, index):
        # grab inputs
        support = batch['support']
        query = batch['query']

        # grab targets
        proto_targets = batch['targets']
        parent_targets = self._squash_parent_targets(batch)

        # grab predictions
        output = self(support, query)
        parent_probits = self._squash_parent_probits(output)

        # compute our prototype loss
        proto_loss = self.criterion(output['distances'], proto_targets)
        parent_loss = self.criterion(parent_probits, parent_taregets)

        loss = (1 - self.alpha) * proto_loss + self.alpha * parent_loss

        output['proto_loss'] = proto_loss
        output['parent_loss'] = parent_loss
        output['loss'] = loss

        return output

    def _log_metrics(self, output, stage='train'):
        self.log(f'loss/{stage}', output['loss'].detach().cpu())
        self.log(f'proto_loss/{stage}', output['loss'].detach().cpu())
        self.log(f'parent_loss/{stage}', output['loss'].detach().cpu())

    def training_step(self, batch, index):
        """Performs one step of training"""
        output = self._main_step(batch, index)
        self._log_metrics(output, stage='train')

        return output['loss']

    def validation_step(self, batch, index):
        """Performs one step of validation"""
        output = self._main_step(batch, index)
        output = batch_detach_cpu(output)
        self._log_metrics(output, stage='val')

    def test_step(self, batch, index):
        """Performs one step of testing"""
        output = self._main_step(batch, index)
        output = batch_detach_cpu(output)
        self._log_metrics(output, stage='test') 

    def configure_optimizer(self):
        """Configure optimizer for training"""
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":

    model = ProtoNet(parents=['strings', 'winds', 'percussion', 'electric'])
    logging.info(model)