"""model.py - model definition"""

from functools import reduce
from operator import __add__
import argparse
import logging
from collections import OrderedDict
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, learning_rate: float, ):
        """ flat protonet for now"""
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # self.example_input_array = torch.zeros((1, 1, 128, 199))
        self.backbone = Backbone()
        backbone_dims = self._get_backbone_shape()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser 

        parser.add_argument('--learning_rate', type=float, default=0.0003,
                            help='learning rate for training. will be decayed using MultiStepLR')

        return parser
    
    def _get_backbone_shape(self):
        # do a dummy forward pass through the
        # embedding model to get the output shape
        i = torch.randn((1, 1, 128, 199))
        o = self.backbone(i)
        return o.shape

    def _forward(self, x):
        """forward pass through the backbone model, as well as the 
        coarse grained classifier. Returns the output, weighed embedding, 
        as well as the probits
        """
        # input should be shape (b, c, f, t)
        x = self.backbone(x)

        return x

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
        d_batch, d_cls, d_k, _, d_frq, d_t = support.shape
        support = support.view(d_batch*d_cls*d_k, 1, d_frq, d_t)
        support = self._forward(support)
        
        support = support.view(d_batch, d_cls, d_k, -1) # expand back
        # support_probits = support_probits.view(d_batch, d_cls, d_k, -1) # expand probits as well

        # take mean over d_k to get the prototypes
        prototypes = support.mean(dim=2, keepdim=False)

        # get query embeddings
        d_batch, q_dim, _, d_frq, d_t = query.shape
        query = query.view(d_batch * q_dim, 1, d_frq, d_t)
        query = self._forward(query)

        query = query.view(d_batch, q_dim, -1)
        # query_probits = query_probits.view(d_batch, q_dim, -1)

        # by here, query should be shape (b, q, n)
        # and prototypes should be shape (b, c, n)

        # compute euclidean distances
        # output shoudl be shape (b, q, c)
        # so that the row vectors are the logits for the classes 
        # for each query, in the batch
        dists = torch.cdist(query, prototypes, p=2)
        return {'distances': dists, }
                # 'query_probits': query_probits, 
                # 'support_probits': support_probits}

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
        proto_targets = batch['target']

        # grab predictions
        output = self(support, query)

        loss = self.criterion(output['distances'], proto_targets)

        output['pred'] = torch.argmax(output['distances'], dim=-1, keepdim=False)
        output['loss'] = loss

        output.update(batch)

        return output

    def _log_metrics(self, output, stage='train'):
        self.log(f'loss/{stage}', output['loss'].detach().cpu())

        # prepare predictions and targets
        pred = output['pred'].view(-1)
        target = output['target'].view(-1)

        breakpoint()
        from pytorch_lightning.metrics.functional import accuracy,\
                                                     f_beta, auroc
        self.log(f'accuracy/{stage}', accuracy(pred, target))
        self.log(f'f1/{stage}', f_beta(pred, target, beta=1))
        self.log(f'auroc/{stage}', auroc(pred, target))

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

    def configure_optimizers(self):
        """Configure optimizer for training"""
        optimizer = torch.optim.Adam(
            # add this lambda so it doesn't crash if part of the model is frozen
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-5)
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.MultiStepLR(
        #         optimizer,
        #         milestones=[0.5 * self.max_epochs,
        #                     0.75 * self.max_epochs],
        #         gamma=0.1)
        # }
        return [optimizer]#, [scheduler]

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

def create_sparse_layers(root_d: int, parents: List[List[str]]):
    current_dim = root_d
    layers = nn.ModuleList()
    classifiers = nn.ModuleList()

    for parent_layer in parents:
        partition_dim = current_dim // len(parents)

        # create a new ModuleDict, where the current sparse layer will reside
        layer = nn.ModuleDict(OrderedDict(
            [(name, nn.Linear(current_dim, partition_dim)) for name in parent_layer]
        ))
        layers.append(layer)
        layers_output_dim = len(parent_layer) * partition_dim
        
        # create a classifier that will take the output of the sparse layer and
        # classify it into the list of parents
        classifier = nn.Linear(layers_output_dim, len(parent_layer))
        classifiers.append(classifier)

        current_dim = layers_output_dim

    return layers, classifiers

def all_unique(lst: List[str]):
    return len(lst) == len(set(lst))

if __name__ == "__main__":
    model = ProtoNet(parents=['strings', 'winds', 'percussion', 'electric'])
    logging.info(model)
