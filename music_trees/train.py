"""train.py - model training"""
import music_trees as mt
from embviz.logger import EmbeddingSpaceLogger

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from str2bool import str2bool
import termtables as tt

MAX_EPISODES = 60000
NUM_VAL_EPISODES = 300
VAL_CHECK_INTERVAL = 100
GRAD_CLIP = 1


def train(args):

    # setup transforms
    audio_tfm = mt.preprocess.LogMelSpec(hop_length=128, win_length=512)
    episode_tfm = mt.preprocess.EpisodicTransform()

    # set up data
    datamodule = mt.data.MetaDataModule(name=args.dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        n_episodes=MAX_EPISODES,
                                        n_class=args.n_class,
                                        n_shot=args.n_shot,
                                        n_query=args.n_query,
                                        audio_tfm=audio_tfm,
                                        epi_tfm=episode_tfm)

    # set up model
    taxonomy = mt.utils.data.load_entry(
        mt.ASSETS_DIR/'taxonomies'/'base-taxonomy.yaml', 'yaml')
    tree = mt.tree.MusicTree.from_taxonomy(taxonomy)
    model = mt.models.prototree.ProtoTree(tree, depth=args.depth)
    task = mt.models.core.ProtoTask(model, learning_rate=args.learning_rate)

    # logging
    from pytorch_lightning.loggers import TestTubeLogger
    logger = TestTubeLogger(save_dir=mt.RUNS_DIR, name=args.name,
                            version=args.version)
    exp_dir = Path(logger.save_dir) / logger.name / \
        f"version_{logger.experiment.version}"
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
                                    monitor='loss/val', save_top_k=3, mode='min')
    callbacks.append(ckpt_callback)

    # early stop
    from pytorch_lightning.callbacks import EarlyStopping
    callbacks.append(EarlyStopping(
        monitor='loss/val', mode='min', patience=20))

    # get the best path to the model if it exists
    if ckpt_callback.best_model_path == '':
        best_model_path = None
    else:
        best_model_path = ckpt_callback.best_model_path

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        precision=16,
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

    if not args.test:
        trainer.fit(task, datamodule=datamodule)
    else:
        results = trainer.test(task, datamodule=datamodule)


if __name__ == '__main__':
    """Parse command-line arguments"""
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
    parser.add_argument('--test', type=str2bool, default=False,
                        required=False)

    # add datamodule arguments
    parser = mt.data.MetaDataModule.add_argparse_args(parser)

    # add model arguments
    parser = mt.models.prototree.ProtoTree.add_model_specific_args(parser)
    parser = mt.models.core.ProtoTask.add_model_specific_args(parser)

    args = parser.parse_args()

    train(args)
