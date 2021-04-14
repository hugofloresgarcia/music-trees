from embviz.logger import EmbeddingSpaceLogger
from music_trees.models.core import load_model

from argparse import Namespace

import torch
import pytorch_lightning as pl
from music_trees.utils.train import batch_detach_cpu


class MetaTask(pl.LightningModule):

    def __init__(self, hparams: Namespace):
        """Training and logging code
        for both meta-learning and regular classification
        tasks.

        one of the required hparams is hparams.model_name.
        It should be the name of a model. See the list
        of available models at mt.models.core.load_model().

        to access the model, use the `.model` attribute.

        Args:
            hparams ([Namespace]): model hyperparameters.
        see MetaTask.add_model_specific_args for a list of
        required hparams
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = load_model(hparams.model_name)
        self.learning_rate = hparams.learning_rate

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser
        parser.add_argument('--model_name', type=str, required=True,
                            help='name of model variant to load. see mt.models.core.load_model')
        parser.add_argument('--learning_rate', type=float, default=0.003,
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

        val_check: bool = output['index'] % self.trainer.val_check_interval == 0

        # grab the list of all tasks
        tasks = output['tasks']
        for task in tasks:
            self.log_classification_task(task, stage)

            # only do the dim reduction every so often
            if val_check:
                self.text_log_task(task, stage)

        if val_check:
            self.visualize_embedding_space(output, stage)

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