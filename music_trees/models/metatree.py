import music_trees as mt

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MetaTree(pl.LightningModule):

    def __init__(self):
        self.is_meta = True
        pass

    def forward(self, x, n_q: int, n_c: int, n_k: int):
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
        dists = torch.cdist(x_q.unsqueeze(0), x_p.unsqueeeze(0), p=2)

        # remove batch dim for dists
        assert dists.shape[0] == 1
        dists = dists[0]

        output['proto_task'] = {
            'distances': -dists,
            'support_embedding': x_s,
            'query_embedding': x_q,
            'prototype_embedding': x_p,
            'tag': 'proto'
        }
        return output

    def compute_multitask_meta_losses(self, output: dict, episode: dict):
        """ 
        calculates multitask meta losses, 
        given an output dict with a 'metatasks' key. 
        """
        for task in output['metatasks']:

            d_batch = output['proto_task']['distances'].shape[0]
            assert d_batch == 1

            # get the classlist for this episode
            classlist = task['classlist']
            prototypes = output['proto_task']['prototype_embedding'][0]

            # get the list of parents for each of the members in the classlist
            ancestors = [self.tree.get_ancestor(c, 1) for c in classlist]

            # class2ancestors
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
            ancestor_dists = torch.cdist(
                query, ancestor_protos.unsqueeze(0), p=2)

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
        loss = 0*proto_loss + 10 * output['ancestor_task']['loss']

        # TODO: fixme
        output['proto_task']['target'] = proto_targets
        output['proto_task']['classlist'] = output['classlist'][0]
        output['proto_task']['loss'] = proto_loss
        output['proto_task']['pred'] = torch.argmax(
            dists, dim=-1, keepdim=False)
        output['loss'] = loss
        return output
