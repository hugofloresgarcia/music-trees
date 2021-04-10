from music_trees.models.prototree import ProtoTree
import torch
import pytorch_lightning as pl
from music_trees.utils.train import batch_detach_cpu
import music_trees as mt


class ProtoTask(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = ProtoTree(hparams.taxonomy_name, hparams.depth)
        self.learning_rate = hparams.learning_rate

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser
        parser.add_argument('--learning_rate', type=float, default=0.0003,
                            help='learning rate for training. will be decayed using MultiStepLR')

        return parser

    def _main_step(self, batch, index, compute_loss: bool = True):
        # grab predictions
        output = self.model(batch)
        output.update(batch)

        # compute losses
        output = self.model.compute_losses(output)

        # last, add the index for logging
        output['index'] = index

        return output

    def training_step(self, batch, index):
        """Performs one step of training"""
        output = self._main_step(batch, index)
        self.log_metrics(output, stage='train')

        return output['loss']

    def validation_step(self, batch, index):
        """Performs one step of validation"""
        with torch.no_grad():
            output = self._main_step(batch, index)
        output = batch_detach_cpu(output)
        self.log_metrics(output, stage='val')

    def test_step(self, batch, index):
        """Performs one step of testing"""
        with torch.no_grad():
            output = self._main_step(batch, index)
        output = batch_detach_cpu(output)
        self.log_metrics(output, stage='test')

    def eval_step(self, batch, index):
        with torch.no_grad():
            output = self._main_step(batch, index)
        return batch_detach_cpu(output)

    def log_metrics(self, output, stage='train'):
        self.log(f'loss/{stage}', output['loss'].detach().cpu())

        val_check: bool = output['index'] % self.trainer.val_check_interval == 0

        # grab the list of all tasks
        tasks = output['tasks']
        tasks.append(output['proto_task'])
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
        """ visualizes the embedding space for 1 piece of the batch """
        if not hasattr(self, 'emb_loggers'):
            return
        idx = 0
        records = output['records'][idx]
        n_support = output['n_shot'][idx] * output['n_class'][idx]
        n_query = output['n_query'][idx] * output['n_class'][idx]

        # class index to class name
        def gc(i): return output['classlist'][idx][i]

        embeddings = [e for e in output['proto_task']['support_embedding'][idx]] + \
            [e for e in output['proto_task']['query_embedding'][idx]]
        labels = [r['label'] for r in records]
        metatypes = ['support'] * n_support + ['query'] * n_query
        audio_paths = [r['audio_path'] for r in records]

        # grab the prototypes now
        for p_emb, label in zip(output['proto_task']['prototype_embedding'][idx], output['classlist'][idx]):
            embeddings.append(p_emb)
            labels.append(label)
            metatypes.append('proto')
            audio_paths.append(None)

        embeddings = torch.stack(embeddings).detach().cpu().numpy()
        self.emb_loggers[stage].add_step(step_key=str(self.global_step), embeddings=embeddings, symbols=metatypes,
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
