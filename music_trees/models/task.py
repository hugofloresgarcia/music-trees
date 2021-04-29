from embviz.logger import EmbeddingSpaceLogger
from music_trees.models.protonet import HierarchicalProtoNet
from music_trees.utils.train import batch_detach_cpu

from argparse import Namespace
import itertools

import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    if len(class_names) > 20:
        figure = plt.figure(figsize=(32, 32))
    else:
        figure = plt.figure(figsize=(12, 12))
    plt.imshow(cm, cmap='viridis')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=65)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "black" if cm[i, j] > threshold else "white"
        plt.text(j, i, "{:0.2f}".format(
            cm[i, j]), horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


class MetaTask(pl.LightningModule):

    def __init__(self, args: Namespace):
        """Training and logging code
        for both meta-learning and regular classification
        tasks.

        one of the required args is args.model_name.
        It should be the name of a model. See the list
        of available models at mt.models.task.load_model().

        to access the model, use the `.model` attribute.

        Args:
            args ([Namespace]): model hyperparameters.

        see MetaTask.add_model_specific_args for a list of
        required args
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = self.load_model(args)
        self.learning_rate = args.learning_rate

    def load_model(self, args):
        """ loads the actual model object"""
        if args.model_name.lower() == 'hprotonet':
            model = HierarchicalProtoNet(args)
        else:
            raise ValueError(f"invalid model name: {args.model_name}")

        return model

    @ staticmethod
    def load_model_parser(parser, args):
        """
        depending on the model_name provided by the user,
        this will finish adding model specific Argparse args for the
        model that this task object holds
        """
        if args.model_name.lower() == 'hprotonet':
            parser = HierarchicalProtoNet.add_model_specific_args(parser)
        else:
            raise ValueError(f"invalid model name: {args.model_name}")

        return parser

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser
        parser.add_argument('--model_name', type=str, required=True,
                            help='name of model variant to load. see mt.models.task.load_model_parser')
        parser.add_argument('--learning_rate', type=float, default=0.03,
                            help='learning rate for training. will be decayed using MultiStepLR')

        return parser

    def _main_step(self, episode: dict, index):
        """
        performs a forward pass through the model,
        and computes the loss, calling self.model.compute_losses.
        """
        # grab predictions
        output = self.model(episode)
        output.update(episode)

        # compute losses
        output = self.model.compute_losses(episode, output)

        # last, add the index for logging
        output['index'] = index

        return output

    def training_step(self, episode: dict, index):
        """Performs one step of training"""
        output = self._main_step(episode, index)
        self.log_metrics(output, stage='train')

        return output['loss']

    def validation_step(self, episode: dict, index):
        """Performs one step of validation"""
        with torch.no_grad():
            output = self._main_step(episode, index)
        output = batch_detach_cpu(output)
        self.log_metrics(output, stage='val')

    def test_step(self, episode: dict, index):
        """Performs one step of testing"""
        with torch.no_grad():
            output = self._main_step(episode, index)
        output = batch_detach_cpu(output)
        self.log_metrics(output, stage='test')

    def eval_step(self, episode: dict, index):
        """ same as test_step, but doesn't call
        log_metrics(), as evaluation metrics are
        computed separately. """
        with torch.no_grad():
            output = self._main_step(episode, index)
        return batch_detach_cpu(output)

    def log_metrics(self, output: dict, stage='train'):
        """ given an output dict,
        logs metrics and embedding spaces to tensorboard and to
        embviz, respectively.

        output should have:
            loss: tensor that contains the main loss
            index: episode index
            tasks (List[dict]): a list of classification tasks,
                where each task is a dictionary with fields:
                    is_meta: bool indicating whether this is a meta task or
                                regular classification
                    classlist: list of classes in task
                    pred: predictions as a 1d array
                    target: targets as a 1d array
                    tag: a unique name for the task
                    n_shot, n_class, n_query (see dataset)
        """
        self.log(f'loss/{stage}', output['loss'].detach().cpu())

        val_check: bool = stage == 'val' and output['index'] == 0

        # grab the list of all tasks
        tasks = output['tasks']
        for task in tasks:
            task['target'] = task['target'].detach().cpu()
            task['pred'] = task['pred'].detach().cpu()

            self.log_classification_task(task, stage)

            # only do the dim reduction every so often
            if val_check:
                self.log_confusion_matrix(task, stage)
                self.text_log_task(task, stage)

        # if val_check:
        #     self.visualize_embedding_space(output, stage)

    def log_classification_task(self, task: dict, stage: str):
        """ log task metrics as scalar"""
        from pytorch_lightning.metrics.functional import accuracy, precision_recall
        from pytorch_lightning.metrics.functional.f_beta import f1

        # NOTE: assume a fixed num_classes across episodes
        num_classes = len(task['classlist'])
        self.log(f'accuracy/{task["tag"]}/{stage}',
                 accuracy(task['pred'], task['target']))
        self.log(f'f1/{task["tag"]}/{stage}', f1(task['pred'], task['target'],
                 num_classes=num_classes, average='weighted'))
        self.log(f'loss-weighted/{task["tag"]}/{stage}',
                 task['loss'] * task['loss_weight'])
        self.log(f'loss-unweighted/{task["tag"]}/{stage}',
                 task['loss'])

    def log_confusion_matrix(self, task: dict, stage: str):
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(
            task['target'], task['pred'], normalize='true')
        fig = plot_confusion_matrix(conf_matrix, task['classlist'])

        self.logger.experiment.add_figure(
            f'{task["tag"]}/{stage}', fig, self.global_step)

    def text_log_task(self, task: dict, stage: str):
        """ log a task via a markdownish table"""
        # class index to class name
        def gc(i): return task['classlist'][i]

        example = f"# Episode \n"
        example += f"\tclasslist: {task['classlist']}\n\n"
        example += f"\t{'prediction':<45}{'target':<45}\n\n"

        for p, t in zip(task['pred'],
                        task['target']):
            example += f"\t{gc(p):<35}{gc(t):<35}\n"

        self.logger.experiment.add_text(
            f'example/{task["tag"]}/{stage}', example, self.global_step)

    def visualize_embedding_space(self, output: dict, stage: str):
        """ visualizes the embedding space for a single episode """
        if not hasattr(self, 'emb_loggers'):
            self.emb_loggers = {}

        records = output['records']
        n_support = output['n_shot'] * output['n_class']
        n_query = output['n_query'] * output['n_class']

        # class index to class name
        def gc(i): return output['classlist'][i]

        metatasks = [t for t in output['tasks'] if t['is_meta']]

        for metatask in metatasks:
            embeddings = [e for e in metatask['support_embedding']] + \
                [e for e in metatask['query_embedding']]

            labels = [r['label'] for r in records]
            if 'label_dict' in metatask:
                labels = [metatask['label_dict'][l] for l in labels]

            metatypes = ['support'] * n_support + ['query'] * n_query
            audio_paths = [r['audio_path'] for r in records]

            # also visualize the prototypes, though without audio
            for p_emb, label in zip(metatask['prototype_embedding'], metatask['classlist']):
                embeddings.append(p_emb)
                labels.append(label)
                metatypes.append('proto')
                audio_paths.append(None)

            embeddings = torch.stack(embeddings).detach().cpu().numpy()

            # one logger per metatask, per partition (train, val, test)
            logger_key = f'{metatask["tag"]}-{stage}'
            if logger_key not in self.emb_loggers:
                self.emb_loggers[logger_key] = EmbeddingSpaceLogger(
                    self.exp_dir / 'embeddings' / logger_key)

            self.emb_loggers[logger_key].add_step(step_key=str(self.global_step), embeddings=embeddings, symbols=metatypes,
                                                  labels=labels, metadata={'audio_path': audio_paths}, title='meta space')

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
                patience=500,
                verbose=True,
            ),
            'interval': 'step',
            'frequency': 1,
            'monitor': 'loss/train'
        }
        return [optimizer], [scheduler]
