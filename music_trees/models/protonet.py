from music_trees.models.backbone import Backbone
import music_trees as mt

from typing import List, OrderedDict, Dict
from argparse import Namespace
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_all_query_records(episode: dict):
    return [e for e in episode['records']
            [episode['n_shot']*episode['n_class']:]]


def get_target_tensor(query_records: List[dict], classlist: List[str]):
    return torch.tensor([classlist.index(e['label'])
                         for e in query_records])


def loss_weight_fn(decay, height):
    return torch.exp(-torch.tensor(decay*height).float())


class HierarchicalProtoNet(nn.Module):
    """
    A regular protonet with a  hierarchical prototypical loss.

    This model has no parameters, and just
    takes care of doing the protonet math

    given output created by the backbone model,
    This will calculate the class prototypes
    for ancestor nodes up a tree, and compute a
    prototype-net loss for each level of a tree.

    if height == 0, just computes a regular prototypical net loss
    """

    def __init__(self, args: Namespace):
        super().__init__()

        self.backbone = Backbone()
        self._backbone_shape = self._get_backbone_shape()
        d_backbone = self._backbone_shape[-1]

        self.linear_proj = nn.Linear(d_backbone, args.d_root)

        assert args.height >= 0
        self.height = args.height

        taxonomy = mt.utils.data.load_entry(
            mt.ASSETS_DIR/'taxonomies'/f'{args.taxonomy_name}.yaml', 'yaml')
        self.tree = mt.tree.MusicTree.from_taxonomy(taxonomy)

        self.loss_decay = 2

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser
        parser.add_argument('--d_root', type=int, default=128,
                            help="root dimension")
        parser.add_argument('--height', type=int, default=0,
                            help="height of prototypical loss")
        parser.add_argument('--taxonomy_name', type=str, default='joint-taxonomy',
                            help="name of taxonomy file. must be in \
                                    music_trees/assets/taxonomies")
        return parser

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

    def forward(self, episode: dict):
        """ 
        forward pass through prototypical net
        will output a single metatask, wrapped in a list. 
        """
        # after collating, x should be
        # shape (batch, n_c*n_k + n_q, -1)
        x = episode['x']

        # forward pass through backbone
        x = self.root_forward(x)

        n_k = episode['n_shot']
        n_c = episode['n_class']
        x_q = episode['n_query']

        # separate support and query set
        x_s = x[:n_k*n_c, :]
        x_s = x_s.view(n_c, n_k, -1)

        x_q = x[n_k*n_c:, :]

        # take mean over d_k to get the prototypes
        x_p = x_s.mean(dim=1, keepdim=False)
        x_s = x_s.view(n_c * n_k, -1)

        # compute euclidean distances between query and prototypes
        # this needs a batch dim, so need to unsqueeze
        # output should be shape (b, q, c)
        # so that the row vectors are the logits for the classes
        # for each query, in the batch
        dists = torch.cdist(x_q.unsqueeze(0), x_p.unsqueeze(0), p=2)

        # remove batch dim for dists
        assert dists.shape[0] == 1
        dists = dists[0]

        metatask = {
            'is_meta': True,
            'classlist': episode['classlist'],
            'distances': -dists,
            'support_embedding': x_s,
            'query_embedding': x_q,
            'prototype_embedding': x_p,
            'tag': 'protonet'
        }

        return {'tasks': [metatask]}

    def get_ancestor_classlist(self, classlist: List[str], height: int):
        # get a 1-to-1 mapping from the parents to ancestors
        # get the list of parents for each of the members in the classlist
        ancestors = [self.tree.get_ancestor(c, height) for c in classlist]

        # note that ancestor_classlist is a UNIQUE set of strings,
        # and should thus be used to generate numeric labels for the ancestors
        ancestor_classlist = list(set(ancestors))

        # map from fine grained classes to our ancestors
        c2a = OrderedDict([(c, a) for c, a in zip(classlist, ancestors)])

        return ancestor_classlist, c2a

    def get_ancestor_targets(self, episode: dict, ancestor_classlist: List[str],
                             class2ancestor: Dict[str, str]):
        """
        creates an UNTYPED (cpu, double) tensor with numeric labels
        for ancestors
        """
        # grab all the query records
        query_records = get_all_query_records(episode)

        # grab all the labels from the query records
        query_labels = [e['label'] for e in query_records]

        # map all_query_labels to their respective ancestor
        # target (as numeric targets for cross_entropy_loss)
        ancestor_targets = [ancestor_classlist.index(class2ancestor[l])
                            for l in query_labels]
        ancestor_targets = torch.tensor(ancestor_targets)
        return ancestor_targets

    def compute_ancestor_prototypes(self, episode: dict, metatask: dict, height: int):
        """ given an episode"""
        # get the classlist for this episode
        classlist = metatask['classlist']
        prototypes = metatask['prototype_embedding']

        # ordered dictionary of prototypes
        # we will use this to group prototypes by ancestor later
        protos = OrderedDict([(n, t)
                              for n, t in zip(classlist, prototypes)])

        # get the unique list of ancestors,
        # as well as dict w format class: ancestor
        ancestor_classlist, c2a = self.get_ancestor_classlist(
            classlist, height)

        # store ancestor prototypes here
        ancestor_protos = OrderedDict()
        for classname in classlist:
            # get this class's ancestor
            ancestor = c2a[classname]

            # if we have already registered this ancestor,
            # it means we already grabbed the relevant tensors for it,
            # so skip it
            if ancestor in ancestor_protos:
                continue

            # otherwise, grab all the tensors that share this ancestor
            tensor_stack = torch.stack(
                [p for n, p in protos.items() if c2a[n] == ancestor])

            # take the prototype of the prototypes!
            ancestor_proto = tensor_stack.mean(dim=0, keepdim=False)

            # register this prototype with the rest of ancestor protos
            ancestor_protos[ancestor] = ancestor_proto

        # convert the OrderedDict into a tensor for our loss function
        ancestor_protos = torch.stack(list(ancestor_protos.values()))

        return ancestor_protos

    def compute_leaf_loss(self, episode: dict, metatask: dict):
        # compute loss
        classlist = metatask['classlist']
        dists = metatask['distances']

        query_records = get_all_query_records(episode)
        target = get_target_tensor(
            query_records, classlist).type_as(dists).long()

        loss = F.cross_entropy(dists, target)

        metatask['target'] = target
        metatask['loss'] = loss
        metatask['pred'] = torch.argmax(dists, dim=-1, keepdim=False)
        metatask['loss'] = loss
        metatask['loss_weight'] = loss_weight_fn(
            self.loss_decay, 0).type_as(loss)
        return metatask

    def compute_ancestor_losses(self, episode: dict, metatask: dict):
        """ 
        calculates multitask meta losses, 
        given a metatask dict which contains:
            prototype_embedding: the leaf node prototypes
            classlist: list of leaf node classnames

        """
        # get the classlist for this episode
        classlist = metatask['classlist']

        ancestor_metatasks = []
        for height in range(1, self.height+1):
            # get the unique list of ancestors,
            # as well as dict w format class: ancestor
            ancestor_classlist, c2a = self.get_ancestor_classlist(
                classlist, height)

            ancestor_protos = self.compute_ancestor_prototypes(
                episode, metatask, height)

            # NOW, let's obtain coarse grained labels for
            # the entire query set
            ancestor_targets = self.get_ancestor_targets(episode, ancestor_classlist,
                                                         c2a).type_as(ancestor_protos).long()

            # grab the query embedding output by ProtoNet
            query = metatask['query_embedding']

            # compute query-prototype distances
            # when using cdist, make sure to unsqueeze a batch dimension
            ancestor_dists = torch.cdist(
                query.unsqueeze(0), ancestor_protos.unsqueeze(0), p=2)

            # remember to remove batch dim from output distances
            ancestor_dists = ancestor_dists.squeeze(0)

            loss = F.cross_entropy(ancestor_dists, ancestor_targets.view(-1))
            ancestor_task = {
                'is_meta': True,
                'classlist': ancestor_classlist,
                'label_dict': c2a,
                'distances': -ancestor_dists,
                'pred': torch.argmax(-ancestor_dists, dim=-1, keepdim=False),
                'prototype_embedding': ancestor_protos,
                'query_embedding': query,
                'support_embedding': metatask['support_embedding'],
                'target': ancestor_targets,
                'tag': 'ancestor-proto',
                'loss': loss,
                'loss_weight': loss_weight_fn(self.loss_decay, height).type_as(loss)
            }

            # to keep the coarse-to-fine scheme,
            # insert the ancestor task at the beginning of the list
            ancestor_metatasks.insert(0, ancestor_task)

        return ancestor_metatasks

    def compute_losses(self, episode: dict, output: dict):
        """ 
        main entry point. calculates both the leaf losses
        and ancestor losses
        """
        input_task = output['tasks'][0]
        assert input_task['tag'] == 'protonet'

        leaf_task = self.compute_leaf_loss(episode, input_task)
        ancestor_tasks = self.compute_ancestor_losses(episode, input_task)

        # insert metatasks in ascending order
        metatasks = [leaf_task] + ancestor_tasks

        output['loss'] = sum(t['loss']*t['loss_weight']
                             for t in output['tasks'])

        output['tasks'] = metatasks
        return output
