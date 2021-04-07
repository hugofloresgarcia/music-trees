from torch.nn.modules.activation import Tanh
import music_trees as mt
from music_trees.tree import MusicTree

from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


def all_unique(lst: List[str]):
    return len(lst) == len(set(lst))


def odstack(od: OrderedDict, dim: int):
    """given an ordered dict with tensors as values,
    will stack the tensors at the given dim"""
    return torch.stack(list(od.values()), dim=dim)


def odcat(od: OrderedDict, dim: int):
    """given an ordered dict with tensors as values,
    will concat the tensors at the given dim"""
    return torch.cat(list(od.values()), dim=dim)


class LayerTree(nn.Module):

    def __init__(self, root_d: int, tree: MusicTree, depth: int):
        """ creates a stack of layers

        self.layers is a nn.ModuleList[nn.ModuleDict[str: nn.Linear]]
        self.classifiers is a nn.ModuleList[nn.Linear]

        Args:
            root_d (int): [description]
            tree (MusicTree): [description]
            depth (int): [description]
        """
        super().__init__()
        self.root_d = root_d
        self.tree = tree

        nodes = []
        for i in range(2, 2+depth):
            nodes.append([n.uid for n in tree.all_nodes_at_depth(i)])

        self.layers, self.classifiers = self.create_sparse_layers(
            root_d, nodes)

    def forward(self, x):
        """given an input tensor,
        will forward pass through each layer in the tree,
        completing a classification task and getting an embedding.
        will return a list of tasks.
        """
        tasks = []

        x_in = x
        for depth, (layerdict, classifier) in enumerate(zip(self.layers, self.classifiers)):
            # get "expert" embeddings for each node at this level
            embs = OrderedDict([(name, layer(x_in))
                               for name, layer in layerdict.items()])

            # concatenate expert embeddings and compute this layer's class probabilities
            classifier_in = odcat(embs, dim=-1)
            probs = classifier(classifier_in)

            # weigh the expert embeddings with the computed class
            assert probs.shape[-1] == len(embs)
            embs = OrderedDict(
                [(name, emb * torch.softmax(probs[:, idx:idx+1], dim=-1))
                    for idx, (name, emb) in enumerate(embs.items())])

            probs = OrderedDict([(name, p.unsqueeze(-1))
                                for name, p in zip(embs.keys(), probs.t())])

            # this constitutes a new task
            task = {
                'embedding': odcat(embs, dim=-1),
                'probs': odcat(probs, dim=-1),
                'classlist': list(embs.keys()),
                'tag': f'layertree-{depth}'
            }
            tasks.append(task)

            x_in = odcat(embs, dim=-1)

        return tasks

    def find_ancestor_labels(self, labels, ancestor_classlist):
        ancestor_labels = []
        for l in labels:
            ancestor = [
                c for c in ancestor_classlist if self.tree.is_ancestor(c, l)]
            assert len(ancestor) == 1
            ancestor_labels.append(ancestor[0])
        return ancestor_labels

    def compute_losses(self, output: dict, criterion: callable):
        labels = output['labels']
        assert isinstance(labels[0], list)

        # maybe the labels should already be flattened by the time they're here?
        # maybe not
        def flatten_list(l): return [i for s in l for i in s]
        labels = flatten_list(labels)

        # grab the list of tasks, should be a List[Dict]
        tasks = output['tasks']
        for task in tasks:
            # grab the ancestor label targets
            classlist = task['classlist']
            probs = task['probs']

            targets = self.find_ancestor_labels(labels, classlist)
            # we don't want onehots, we want the onehot's argmax
            targets = [torch.argmax(torch.tensor(mt.utils.data.get_one_hot(a, classlist)), dim=-1)
                       for a in targets]
            targets = torch.stack(targets).type_as(probs).long()
            task['target'] = targets

            loss = criterion(probs, targets)

            task['loss'] = loss

            task['probs'] = torch.softmax(probs, dim=-1)
            task['pred'] = torch.argmax(task['probs'], dim=-1, keepdim=False)

        return output

    @staticmethod
    def create_sparse_layers(root_d: int, nodes: List[List[str]]):
        layers = nn.ModuleList()
        classifiers = nn.ModuleList()

        current_dim = root_d
        for parent_layer in nodes:
            partition_dim = current_dim // len(parent_layer)

            # create a new ModuleDict, where the current sparse layer will reside
            layer = nn.ModuleDict(OrderedDict(
                [(name,
                    nn.Sequential(
                        nn.Linear(current_dim, partition_dim, bias=True),
                        nn.ReLU(),
                    )
                  )
                 for name in parent_layer]
            ))
            layers.append(layer)
            layers_output_dim = len(parent_layer) * partition_dim

            # create a classifier that will take the output of the sparse layer and
            # classify it into the list of parents
            classifier = nn.Linear(layers_output_dim, len(parent_layer))
            classifiers.append(classifier)

            current_dim = layers_output_dim

        return layers, classifiers
