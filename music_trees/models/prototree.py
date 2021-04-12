from typing import OrderedDict
import music_trees as mt
from music_trees.tree import MusicTree
from music_trees.models.layer_tree import LayerTree, odstack, odcat
from music_trees.models.backbone import Backbone

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoTree(pl.LightningModule):

    def __init__(self, taxonomy_name: str, depth: int):
        super().__init__()

        taxonomy = mt.utils.data.load_entry(
            mt.ASSETS_DIR/'taxonomies'/f'{taxonomy_name}.yaml', 'yaml')
        tree = mt.tree.MusicTree.from_taxonomy(taxonomy)
        self.tree = tree

        self.backbone = Backbone()
        self._backbone_shape = self._get_backbone_shape()
        d_backbone = self._backbone_shape[-1]

        d_root = d_backbone // 8
        self.linear_proj = nn.Linear(d_backbone, d_root)

        self.layer_tree = LayerTree(d_root, tree, depth=depth)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser
        parser.add_argument('--depth', type=int, default=2,
                            help='depth of the LayerTree.')
        parser.add_argument('--taxonomy_name', type=str,
                            default='joint-taxonomy')

        return parser

    def _get_backbone_shape(self):
        # do a dummy forward pass through the
        # embedding model to get the output shape
        i = torch.randn((1, 1, 128, 199))
        o = self.backbone(i)
        return o.shape

    def _forward_one(self, x):
        """forward pass through the backbone model, as well as the
        coarse grained classifier. Returns the output, weighed embedding,
        as well as the probits
        """
        # input should be shape (b, c, f, t)
        backbone_embedding = self.backbone(x)
        root_embedding = self.linear_proj(backbone_embedding)

        output = {}
        output['tasks'] = self.layer_tree(root_embedding)

        # add backbone embedding to output
        output['backbone'] = root_embedding

        if len(output['tasks']) > 0:
            output['embedding'] = output['tasks'][-1]['embedding']
            # output['embedding'] = root_embedding
        else:
            output['embedding'] = root_embedding

        return output

    def forward(self, episode):
        # after collating, x should be
        # shape (batch, n_c*n_k + n_q, -1)
        x = episode['x']

        # get batch dim
        n_b = x.shape[0]
        n_k = episode['n_shot'][0]
        n_q = episode['n_query'][0]
        n_c = episode['n_class'][0]

        # flatten episode
        x = x.view(n_b * n_c * (n_k + n_q), *x.shape[2:])

        # forward pass everything
        output = self._forward_one(x)
        x = output['embedding']

        # expand back to batch
        x = x.view(n_b, n_c * (n_k + n_q), -1)

        # separate support and query set
        x_s = x[:, :n_k*n_c, :]
        x_s = x_s.view(n_b, n_c, n_k, -1)
        x_q = x[:, n_k*n_c:, :]

        # take mean over d_k to get the prototypes
        x_p = x_s.mean(dim=2, keepdim=False)
        x_s = x_s.view(n_b, n_c * n_k, -1)

        # compute euclidean distances
        # output shoudl be shape (b, q, c)
        # so that the row vectors are the logits for the classes
        # for each query, in the batch
        dists = torch.cdist(x_q, x_p, p=2)

        output['proto_task'] = {
            'distances': -dists,
            'support_embedding': x_s,
            'query_embedding': x_q,
            'prototype_embedding': x_p,
            'tag': 'proto'
        }
        return output

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
        # compute loss
        proto_targets = output['proto_target'].view(-1)

        dists = output['proto_task']['distances']
        b, q, _ = dists.shape
        dists = dists.view(b*q, -1)
        proto_loss = F.cross_entropy(
            dists, proto_targets)

        output = self.layer_tree.compute_losses(output, F.cross_entropy)
        tree_losses = [t['loss'] for t in output['tasks']]
        # loss = proto_loss + 1 * sum(tree_losses)
        loss = proto_loss + 10 * output['ancestor_task']['loss']

        # TODO: fixme
        output['proto_task']['target'] = proto_targets
        output['proto_task']['classlist'] = output['classlist'][0]
        output['proto_task']['loss'] = proto_loss
        output['proto_task']['pred'] = torch.argmax(
            dists, dim=-1, keepdim=False)
        output['loss'] = loss
        return output
