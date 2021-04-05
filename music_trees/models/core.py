"""model.py - model definition
#TODO: get the logging and visualization logic outta here
and also the convblock and backbone modules
"""
import music_trees as mt
from music_trees.tree import MusicTree
from music_trees.utils.train import batch_detach, batch_cpu, \
    batch_detach_cpu
from music_trees.models.backbone import Backbone

import logging
from collections import OrderedDict
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        for i in range(2,2+depth):
            nodes.append([n.uid for n in tree.all_nodes_at_depth(i)])

        self.layers, self.classifiers = self.create_sparse_layers(
            root_d, nodes)

    def forward(self, x):
        embeddings = []
        predictions = []

        x_in = x
        # breakpoint()
        for layerdict, classifier in zip(self.layers, self.classifiers):
            # get "expert" embeddings for each node at this level
            embs = OrderedDict([(name, layer(x_in))
                               for name, layer in layerdict.items()])

            # concatenate expert embeddings and compute this layer's class probabilities
            classifier_in = odcat(embs, dim=-1)
            probs = classifier(classifier_in)

            # weigh the expert embeddings with the computed class 
            assert probs.shape[-1] == len(embs)
            embs = OrderedDict(
                [(name, emb * probs[:, idx:idx+1]) for idx, (name, emb) in enumerate(embs.items())])

            probs = OrderedDict([(name, p.unsqueeze(-1))
                                for name, p in zip(embs.keys(), probs.t())])

            embeddings.append(embs)
            predictions.append(probs)

            x_in = odcat(embs, dim=-1)

        return {'embeddings': embeddings, 'predictions': predictions}

    def find_ancestor_labels(self, labels, ancestor_classlist):
        ancestor_labels = []
        for l in labels:
            ancestor = [
                c for c in ancestor_classlist if self.tree.is_ancestor(c, l)]
            assert len(ancestor) == 1
            ancestor_labels.append(ancestor[0])
        return ancestor_labels

    def compute_losses(self, labels: list, output: dict, 
                    criterion: callable):
        # FIX ME WITH A FIXED BATCH SIZE OF 1 EVERYWHERE
        assert len(labels) == 1
        labels = labels[0]
        # grab the list of predictions. should be an List[OrderedDict]
        preds = output['predictions']
        
        losses = []
        # go through each prediction dict
        for pred_dict in preds:
            # gather the ancestor label targets
            classlist = list(pred_dict.keys())    
            ancestor_labels = self.find_ancestor_labels(labels, classlist)
            # we don't want onehots, we want the onehot's argmax
            ancestor_labels = [torch.argmax(torch.tensor(mt.utils.data.get_one_hot(a, classlist)), dim=-1)
                                         for a in ancestor_labels]
            ancestor_labels = torch.stack(ancestor_labels)

            ancestor_preds = odcat(pred_dict, -1)
            
            loss = criterion(ancestor_preds, ancestor_labels.type_as(ancestor_preds).long())
            losses.append(loss)
        
        return losses

    @staticmethod
    def create_sparse_layers(root_d: int, nodes: List[List[str]]):
        layers = nn.ModuleList()
        classifiers = nn.ModuleList()

        current_dim = root_d
        for parent_layer in nodes:
            partition_dim = current_dim // len(parent_layer)

            # create a new ModuleDict, where the current sparse layer will reside
            layer = nn.ModuleDict(OrderedDict(
                [(name, nn.Linear(current_dim, partition_dim))
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

class ProtoNet(pl.LightningModule):

    def __init__(self, tree: MusicTree, learning_rate: float, depth: int):
        """ flat protonet for now"""
        super().__init__()
        self.save_hyperparameters('learning_rate', 'depth')
        self.learning_rate = learning_rate

        # self.example_input_array = torch.zeros((1, 1, 128, 199))
        self.backbone = Backbone()
        self._backbone_shape = self._get_backbone_shape()
        d_backbone = self._backbone_shape[-1]

        d_root = d_backbone // 8
        self.linear_proj = nn.Linear(d_backbone, d_root)

        self.layer_tree = LayerTree(d_root, tree, depth=depth)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser
        parser.add_argument('--learning_rate', type=float, default=0.0003,
                            help='learning rate for training. will be decayed using MultiStepLR')
        parser.add_argument('--depth', type=int, default=2,
                            help='depth of the LayerTree.')

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

        output = self.layer_tree(root_embedding)

        # add backbone embedding to output
        output['backbone'] = root_embedding

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
        x = odcat(output['embeddings'][-1], dim=-1)

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

        output.update({'distances': -dists,
                'support_embedding': x_s,
                'query_embedding': x_q,
                'prototype_embedding': x_p})
        return output

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

    def _main_step(self, batch, index):
        # grab targets
        proto_targets = batch['proto_target']

        # grab predictions
        output = self(batch)

        # breakpoint()
        loss = self.criterion(output['distances'], proto_targets)
        tree_losses = self.layer_tree.compute_losses(batch['labels'], output, F.cross_entropy) 
        loss = loss + 0.1*sum(tree_losses)

        output['pred'] = torch.argmax(
            output['distances'], dim=-1, keepdim=False)
        output['loss'] = loss

        output.update(batch)

        # last, add the index for logging
        output['index'] = index

        return output

    def _log_metrics(self, output, stage='train'):
        self.log(f'loss/{stage}', output['loss'].detach().cpu())

        # prepare predictions and targets
        pred = output['pred'].view(-1)
        target = output['proto_target'].view(-1)

        # breakpoint()
        from pytorch_lightning.metrics.functional import accuracy, precision_recall
        from pytorch_lightning.metrics.functional.f_beta import f1

        #NOTE: assume a fixed num_classes across episodes
        num_classes = len(output['classes'][0])
        self.log(f'accuracy/{stage}', accuracy(pred, target))
        self.log(f'f1/{stage}', f1(pred, target,
                 num_classes=num_classes, average='weighted'))

        # only do the dim reduction every so often
        if output['index'] % self.trainer.val_check_interval == 0:
            self.log_single_example(output, stage)
            self.visualize_embedding_space(output, stage)

    def log_single_example(self, output: dict, stage: str):
        """ logs output vs predictions as a table for a slice of the batch"""
        idx = 0
        # class index to class name
        def gc(i): return output['classes'][idx][i]

        example = f"# Episode \n"
        example += f"\tclasslist: {output['classes'][idx]}\n\n"
        example += f"\t{'prediction':<45}{'target':<45}\n\n"

        for p, t in zip(output['pred'][idx],
                        output['proto_target'][idx]):
            example += f"\t{gc(p):<35}{gc(t):<35}\n"

        self.logger.experiment.add_text(
            f'example/{stage}', example, self.global_step)

    def visualize_embedding_space(self, output: dict, stage: str):
        """ visualizes the embedding space for 1 piece of the batch """
        if not hasattr(self, 'emb_loggers'):
            return
        idx = 0
        records = output['records'][idx]
        n_support = output['n_shot'][idx] * output['n_class'][idx]
        n_query = output['n_query'][idx] * output['n_class'][idx]

        # class index to class name
        def gc(i): return output['classes'][idx][i]

        embeddings = [e for e in output['support_embedding'][idx]
                      ] + [e for e in output['query_embedding'][idx]]
        labels = [r['label'] for r in records]
        metatypes = ['support'] * n_support + ['query'] * n_query
        audio_paths = [r['audio_path'] for r in records]

        # grab the prototypes now
        for p_emb, label in zip(output['prototype_embedding'][idx], output['classes'][idx]):
            embeddings.append(p_emb)
            labels.append(label)
            metatypes.append('proto')
            audio_paths.append(None)

        embeddings = torch.stack(embeddings).detach().cpu().numpy()
        self.emb_loggers[stage].add_step(step_key=str(self.global_step), embeddings=embeddings, symbols=metatypes,
                                         labels=labels, metadata={'audio_path': audio_paths}, title='meta space')

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
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=100,
                verbose=True,
            ),
            'interval': 'step',
            'frequency': 1,
            'monitor': 'loss/train'
        }
        return [optimizer], [scheduler]

if __name__ == "__main__":
    model = ProtoNet(parents=['strings', 'winds', 'percussion', 'electric'])
    logging.info(model)
