from music_trees.models.backbone import Backbone
import music_trees as mt

from typing import List, Dict
from argparse import Namespace
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_all_query_records(episode: dict):
    return [e for e in episode['records']
            [episode['n_shot']*episode['n_class']:]]


def get_all_support_records(episode: dict):
    return [e for e in episode['records']
            [:episode['n_shot']*episode['n_class']]]


def get_target_tensor(query_records: List[dict], classlist: List[str]):
    return torch.tensor([classlist.index(e['label'])
                         for e in query_records])


def chunks(l, n):
    """https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks"""
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]


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

        # inserting hack here to match paper
        if self.height > 0:
            self.height += 1

        taxonomy = mt.utils.data.load_entry(
            mt.ASSETS_DIR/'taxonomies'/f'{args.taxonomy_name}.yaml', 'yaml')

        self.tree = mt.tree.MusicTree.from_taxonomy(taxonomy)
        self.tree.even_depth()
        self.tree.show()

        self.loss_weight_fn = args.loss_weight_fn
        self.loss_alpha = args.loss_alpha
        self.loss_beta = args.loss_beta

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser
        parser.add_argument('--d_root', type=int, default=128,
                            help="root dimension")
        parser.add_argument('--height', type=int, default=0,
                            help="height of prototypical loss")
        parser.add_argument('--taxonomy_name', type=str, default='deeper-mdb',
                            help="name of taxonomy file. must be in \
                                    music_trees/assets/taxonomies")
        parser.add_argument('--loss_weight_fn', type=str, default='exp')
        parser.add_argument('--loss_alpha', type=float, default=0.75)
        parser.add_argument('--loss_beta', type=float, default=0.5)
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
        MAX_FP_SIZE = 512
        # maybe this mitigates OOM during evaluation?
        # update: it does!! we're keeping it
        x = torch.cat([self.root_forward(subx)
                       for subx in chunks(x, MAX_FP_SIZE)], dim=0)
        del episode['x']

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
        # I should probably rearrange ancestor_protos here to match ancestor_classlist
        ancestor_protos = [ancestor_protos[a] for a in ancestor_classlist]
        ancestor_protos = torch.stack(ancestor_protos)

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
        metatask['pred'] = torch.argmax(dists, dim=-1, keepdim=False)
        metatask['loss'] = loss
        metatask['include_in_loss'] = True
        metatask['loss_height'] = 0
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
        total_height = self.tree.depth()
        for height in range(1, total_height):
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

            loss = F.cross_entropy(-ancestor_dists, ancestor_targets.view(-1))
            ancestor_task = {
                'classlist': ancestor_classlist,
                'label_dict': c2a,
                'distances': -ancestor_dists,
                'pred': torch.argmax(-ancestor_dists, dim=-1, keepdim=False),
                'prototype_embedding': ancestor_protos,
                'query_embedding': query,
                'support_embedding': metatask['support_embedding'],
                'target': ancestor_targets,
                'tag': f'prototree-h{height}',
                'loss': loss,
                'loss_height':  height - self.height + 1,
                'include_in_loss': (total_height-height) < self.height
            }
            ancestor_metatasks.append(ancestor_task)

        return ancestor_metatasks

    def hierarchy_multi_hot(self, metatasks):
        """
        hierarchy_multi_hot - generate a multihot encoding of the classification task,
        encodes which path of the tree was selected for a predition or target 
        EX. [
                [one-hot height=n, one-hot height=n-1, ..., one-hot height=0,]
                .
                .
                .   
            ]
        """
        # start by getting the size of our multi-hot predictions and targets
        multi_axis_0 = metatasks[0]['target'].shape[0]
        multi_axis_1 = 1

        # find the shape of axis 1
        for task in metatasks:
            multi_axis_1 += len(task['classlist'])

        # create the multihot matrix
        multi_hot_shape = (multi_axis_0, multi_axis_1)

        # hold the one hot encodings
        targets_list = []
        preds_list = []

        # iterating over mettask gives us the task at each height of the tree
        for i, task in enumerate(metatasks):
            # get the current tasks/heights one hot encodings
            # shape: [examples, # of classes at height]
            one_hot_targets = F.one_hot(task['target'])
            # shape: [examples, # of classes at height]
            pred_logits = task['distances']

            # one_hot_targets = task['target'] # shape: [examples, # of classes at height]
            # one_hot_preds = task['pred'] # shape: [examples, # of classes at height]

            targets_list.append(one_hot_targets)
            preds_list.append(pred_logits)

        # use the lists holding one hot encodings to make multihot-encodings
        multi_hot_targets = torch.cat(tuple(targets_list), 1)
        preds_logits = torch.cat(tuple(preds_list), 1)

        loss = F.binary_cross_entropy_with_logits(
            preds_logits.float(), multi_hot_targets.float())
        return loss

    def compute_losses(self, episode: dict, output: dict):
        """
        main entry point. calculates both the leaf losses
        and ancestor losses
        """
        input_task = output['tasks'][0]
        assert input_task['tag'] == 'protonet'

        leaf_task = self.compute_leaf_loss(episode, input_task)
        ancestor_tasks = self.compute_ancestor_losses(episode, input_task)
        metatasks = [leaf_task] + ancestor_tasks
        metatasks = [t for t in metatasks if t['include_in_loss']]

        loss_vec = torch.stack([t['loss'] for t in metatasks])

        if self.height > 0:

            if self.loss_weight_fn == "exp":
                self.loss_weights = torch.exp(
                    -self.loss_alpha * torch.arange(self.height))

                output['loss'] = torch.sum(
                    self.loss_weights.type_as(loss_vec) * loss_vec)

            elif self.loss_weight_fn == "exp-leafheavy":
                self.loss_weights = torch.exp(
                    self.loss_alpha * torch.arange(self.height-1, -1, -1))
                self.loss_weights[0] = torch.tensor(1)

                output['loss'] = torch.sum(
                    self.loss_weights.type_as(loss_vec) * loss_vec)

            elif self.loss_weight_fn == "interp-avg":
                self.loss_weights = torch.ones(
                    self.height) / (self.height-1) * (1 - self.loss_alpha)
                self.loss_weights[0] = 1 * self.loss_alpha

                output['loss'] = self.loss_alpha * loss_vec[0] + \
                    (1 - self.loss_alpha) * torch.mean(loss_vec[1:])

            elif self.loss_weight_fn == "interp-avg-decay":
                # alpha should be 0.5-1.0 (1 for baseline, 0 for all hierarchical)
                # beta should be 0.75, 1, 1.25, (0 for interp-avg)

                #  beta is an exponential decay factor for tree losses
                # alpha is a linear interpolation factor for mixing tree loss with leaf loss
                self.loss_weights = torch.exp(
                    - self.loss_beta * torch.arange(self.height-1, -1, -1)) * (1 - self.loss_alpha)
                self.loss_weights[0] = torch.tensor(1) * self.loss_alpha
                self.loss_weights = self.loss_weights.type_as(loss_vec)

                output['loss'] = self.loss_alpha * loss_vec[0] + \
                    torch.mean(loss_vec[1:] * self.loss_weights[1:])

            elif self.loss_weight_fn == "cross-entropy":
                self.loss_weights = torch.tensor([1, 0, 0, 0])
                output['loss'] = self.hierarchy_multi_hot(metatasks)

            else:
                raise ValueError

        else:
            self.loss_weights = torch.tensor([1, 0, 0, 0])
            output['loss'] = metatasks[0]['loss']

        # insert metatasks in ascending order
        output['tasks'] = metatasks
        output['loss-weights'] = self.loss_weights.detach().cpu()

        return output
