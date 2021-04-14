from typing import List, OrderedDict

from music_trees.models.backbone import Backbone

import torch
import torch.nn as nn


class ModelContainer(nn.Module):

    def __init__(self, d_root: int, head=nn.Module):
        """ 
        metalearning + supervised audio classification model. 

        model components:
            - backbone: backbone CNN (see backbone.py)
            - classification head

        The classification head will receive the root embedding and
        carry out classification tasks. There are two different types
        of classification tasks: metatasks (few shot learning, i.e. protonets)
        and vanilla tasks (classification via maximum likelihood estimation 
                                                aka linear + softmax)

        During a forward pass, the classification heads will receive an 
        episode (dict) as input, and will output a list of dicts, where each
        dict contains all the data related to a single task. 

        That is, a single classification head can carry out multiple classification
        tasks (each task could be a level of a hierarchical tree, for example). 
        """
        super().__init__()
        self.backbone = Backbone()
        self._backbone_shape = self._get_backbone_shape()
        d_backbone = self._backbone_shape[-1]

        self.linear_proj = nn.Linear(d_backbone, d_root)
        self.head = head

    def _get_backbone_shape(self):
        # do a dummy forward pass through the3
        # embedding model to get the output shape
        i = torch.randn((1, 1, 128, 199))
        o = self.backbone(i)
        return o.shape

    def root_forward(self, x):
        """
        forward pass through the backbone model, as well as the
        linear projection. Returns the root embedding as output
        """
        # input should be shape (b, c, f, t)
        backbone_embedding = self.backbone(x)
        root_embedding = self.linear_proj(backbone_embedding)

        return root_embedding

    def forward(self, episode):
        """forward pass through backbone embedding

        Args:
            episode (dict): training episode. 

        Returns:
            dict: output dictionary, with a list of tasks and metatasks
                  completed 
        """
        # after collating, x should be
        # shape (batch, n_c*n_k + n_q, -1)
        x = episode['x']

        # forward pass through backbone
        root = self.root_forward(x)
        episode['root_embedding'] = root

        # now, create an output dict
        output = {'tasks': []}

        # TODO: how will we pipe inputs // outputs
        # amongst the classification heads? or
        # should we just make compount classification heads for that?
        # that sounds like a good idea

        # go through the classification heads
        # pass them the episode itself, as well as the output
        # so far
        tasks = self.head(episode)
        output['tasks'] = tasks

        return output

    def compute_losses(self, episode: dict, output: dict):
        output['tasks'] = self.head.compute_losses(episode, output['tasks'])

        # do the weighted sum of losses]
        output['loss'] = sum(t['loss']*t['loss_weight']
                             for t in output['tasks'])
        return output
