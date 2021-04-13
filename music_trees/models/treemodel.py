from typing import OrderedDict

import music_trees as mt
from music_trees.models.layertree import LayerTree
from music_trees.models.metatree import MetaTree
from music_trees.models.backbone import Backbone

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeModel(pl.LightningModule):

    def __init__(self, taxonomy_name: str, depth: int):
        """ 
        metalearning + supervised audio classification model. 

        # TODO: support a swappable classification frontend
        model components:
            - backbone: backbone CNN (see backbone.py)
            - layertree: 

        combines a layertree (for regular classification tasks) 
        and a metatree (for meta learning tasks)
        """
        raise NotImplementedError
        super().__init__()

        taxonomy = mt.utils.data.load_entry(
            mt.ASSETS_DIR/'taxonomies'/f'{taxonomy_name}.yaml', 'yaml')
        tree = mt.tree.MusicTree.from_taxonomy(taxonomy)
        self.tree = tree

        self.backbone = Backbone()
        self._backbone_shape = self._get_backbone_shape()
        d_backbone = self._backbone_shape[-1]

        self.linear_proj = nn.Linear(d_backbone, d_root)

        self.heads = load_heads()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser
        parser.add_argument('--depth', type=int, default=2,
                            help='depth of the LayerTree.')
        parser.add_argument('--taxonomy_name', type=str,
                            default='joint-taxonomy',
                            help='name of the taxonomy file to use')
        parser.add_argument('--d_root', type=int,
                            default=128,
                            help='root dimension for all tasks')

        return parser

    def _get_backbone_shape(self):
        # do a dummy forward pass through the3
        # embedding model to get the output shape
        i = torch.randn((1, 1, 128, 199))
        o = self.backbone(i)
        return o.shape

    def root_forward(self, x):
        """forward pass through the backbone model, as well as the
        linear projection. Returns the root embedding as output
        """
        # input should be shape (b, c, f, t)
        backbone_embedding = self.backbone(x)
        root_embedding = self.linear_proj(backbone_embedding)

        return root_embedding

    def forward(self, episode):
        # after collating, x should be
        # shape (batch, n_c*n_k + n_q, -1)
        x = episode['x']

        # forward pass through backbone
        root = self._forward_one(x)
        episode['root_embedding'] = root

        # now, create an output dict
        output = {'tasks': [], 'metatasks': []}

        # TODO: how will we pipe inputs // outputs
        # amongst the classification heads? or
        # should we just make compount classification heads for that?
        # that sounds like a good idea

        # go through the classification heads
        # pass them the episode itself, as well as the output
        # so far
        for head in self.heads():
            tasks = head(episode, output)
            if head.is_meta:
                output['metatasks'].extend(tasks)
            else:
                output['tasks'].extend(tasks)

        return outputs

    def compute_multitask_proto_losses(self, output: dict, episode: dict):
        d_batch = output['proto_task']['distances'].shape[0]
        assert d_batch == 1

        # get the classlist for this episode
        classlist = output['classlist'][0]
        prototypes = output['proto_task']['prototype_embedding'][0]

        # get the list of parents for each of the members in the classlist
        ancestors = [self.tree.get_ancestor(c, 1) for c in classlist]

        c2a = OrderedDict([(c, a) for c, a in zip(classlist, ancestors)])

        protos = OrderedDict([(n, t)
                              for n, t in zip(classlist, prototypes)])

        ancestor_protos = OrderedDict()

        for clsname in protos:
            ancestor = c2a[clsname]

            # if we have already registered this ancestor,
            # it means we already grabbed the relevant tensors for it,
            # so skip it
            if ancestor in ancestor_protos:
                continue

            # otherwise, grab all the tensors which share this ancestor
            tensor_stack = torch.stack(
                [p for n, p in protos.items() if c2a[n] == ancestor])

            # take the prototype of the prototypes!
            ancestor_proto = tensor_stack.mean(dim=0, keepdim=False)

            # register this stack with the ancestor protos
            ancestor_protos[ancestor] = ancestor_proto

        all_query_records = episode['records'][0][episode['n_shot'][0]
                                                  * episode['n_class'][0]:]
        all_query_labels = [e['label'] for e in all_query_records]
        ancestor_targets = []
        for l in all_query_labels:
            # add a onehot too
            ancestor = c2a[l]
            onethot = torch.tensor(mt.utils.data.get_one_hot(
                ancestor, list(set(ancestors))))
            ancestor_targets.append(onethot.type_as(ancestor_proto).long())

        ancestor_targets = torch.argmax(torch.stack(
            ancestor_targets), dim=-1, keepdim=False)
        ancestor_protos = torch.stack(list(ancestor_protos.values()))

        query = output['proto_task']['query_embedding'][0].unsqueeze(0)
        ancestor_dists = torch.cdist(query, ancestor_protos.unsqueeze(0), p=2)

        ancestor_dists = ancestor_dists.squeeze(0)
        loss = F.cross_entropy(ancestor_dists, ancestor_targets.view(-1))
        output['ancestor_task'] = {
            'classlist': list(set(ancestors)),
            'distances': -ancestor_dists,
            'pred': torch.argmax(-ancestor_dists, dim=-1, keepdim=False),
            'prototype_embedding': ancestor_protos,
            'query_embedding': query,
            'target': ancestor_targets,
            'tag': 'ancestor-proto',
            'loss': loss
        }
        return output

    def compute_losses(self, output: dict):
