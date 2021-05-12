"""train.py - model training"""
import music_trees as mt
from embviz.logger import EmbeddingSpaceLogger

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

pl.seed_everything(mt.SEED)

MAX_EPISODES = 60000
NUM_VAL_EPISODES = 300
VAL_CHECK_INTERVAL = 150
GRAD_CLIP = 1


def get_exp_dir(name: str, version: int):
    """ get the path to experiment """
    return Path(mt.RUNS_DIR) / name / f"version_{version}"


def train(args, use_ray=False):
    """ train a model """
    # setup transforms
    audio_tfm = mt.preprocess.LogMelSpec(hop_length=mt.HOP_LENGTH,
                                         win_length=mt.WIN_LENGTH)

    # set up data
    kwargs = dict(n_class=args.n_class,
                  n_shot=args.n_shot,
                  n_query=args.n_query,
                  audio_tfm=audio_tfm,)
    tr_kwargs = dict(n_episodes=MAX_EPISODES, **kwargs)
    cv_kwargs = dict(n_episodes=NUM_VAL_EPISODES, **kwargs)
    datamodule = mt.data.MetaDataModule(name=args.dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        tr_kwargs=tr_kwargs,
                                        cv_kwargs=cv_kwargs,
                                        tt_kwargs=cv_kwargs)

    # set up model
    task = mt.models.task.MetaTask(args)

    # logging
    from pytorch_lightning.loggers import TestTubeLogger
    logger = TestTubeLogger(save_dir=mt.RUNS_DIR, name=args.name,
                            version=args.version)
    exp_dir = get_exp_dir(args.name, logger.experiment.version)
    task.exp_dir = exp_dir

    emb_loggers = {part: EmbeddingSpaceLogger(exp_dir / f'{part}-embeddings', n_components=2,
                                              method='tsne') for part in ('train', 'val', 'test')}
    task.emb_loggers = emb_loggers

    # CALLBACKS
    callbacks = []

    # log learning rate
    from pytorch_lightning.callbacks import LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # checkpointing
    from pytorch_lightning.callbacks import ModelCheckpoint
    ckpt_callback = ModelCheckpoint(dirpath=exp_dir / 'checkpoints', filename=None,
                                    monitor='loss/val', save_top_k=1, mode='min')
    callbacks.append(ckpt_callback)

    # early stop
    from pytorch_lightning.callbacks import EarlyStopping
    callbacks.append(EarlyStopping(
        monitor='f1/protonet/val', mode='min', patience=15))

    # get the best path to the model if it exists
    if ckpt_callback.best_model_path == '':
        best_model_path = None
    else:
        best_model_path = ckpt_callback.best_model_path

    # hyperparameter tuning
    if use_ray:
        from ray.tune.integration.pytorch_lightning import TuneReportCallback
        callbacks.append(
            TuneReportCallback({
                "f1/protonet/val": "f1/protonet/val",
            }, on="validation_end"))

        args.progress_bar_refresh_rate = 0

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        auto_lr_find=True,
        max_steps=MAX_EPISODES,
        limit_val_batches=NUM_VAL_EPISODES,
        val_check_interval=VAL_CHECK_INTERVAL,
        callbacks=callbacks,
        logger=logger,
        terminate_on_nan=True,
        resume_from_checkpoint=best_model_path,
        log_gpu_memory=True,
        gpus='0' if torch.cuda.is_available() else None,
        profiler=pl.profiler.SimpleProfiler(
            output_filename=exp_dir / 'profiler-report.txt'),
        gradient_clip_val=GRAD_CLIP,
        deterministic=True,
        num_sanity_val_steps=0,
        move_metrics_to_cpu=True)

    trainer.fit(task, datamodule=datamodule)


def load_parser(known_args=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # add trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)

    # add training script arguments
    parser.add_argument('--name', type=str, required=True,
                        help='name of the experiment')
    parser.add_argument('--version', type=int, required=False,
                        help='version. If not provided, a new version is created')

    # add datamodule arguments
    parser = mt.data.MetaDataModule.add_argparse_args(parser)

    # add model arguments
    parser = mt.models.task.MetaTask.add_model_specific_args(parser)

    # load arguments specific to model
    print('PARSING KNOWN ARGS')
    args, _ = parser.parse_known_args(args=known_args)

    # add model specific arguments for model
    parser = mt.models.task.MetaTask.load_model_parser(parser, args)

    return parser


if __name__ == '__main__':

    parser = load_parser()
    args = parser.parse_args()

    train(args)
