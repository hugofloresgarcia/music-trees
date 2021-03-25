"""train.py - model training"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

import music_trees as mt

MAX_EPOCHS = 100
GRAD_CLIP = 1

N_CLASS = 12
N_SHOT = 4
N_QUERY = 16

def train(args):
   
    # setup transforms
    audio_tfm = mt.preprocess.LogMelSpec(hop_length=128, win_length=512)
    episode_tfm = mt.preprocess.EpisodicTransform(audio_tfm, n_class=N_CLASS, 
                                                  n_shot=N_SHOT, n_query=N_QUERY)

    # set up data
    datamodule = mt.data.MetaDataModule(name=args.dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers, 
                                        n_class=N_CLASS, 
                                        n_shot=N_SHOT, 
                                        n_query=N_QUERY, 
                                        transform=episode_tfm)

    # set up model
    model = mt.model.ProtoNet(
        learning_rate=args.learning_rate)

    # logging
    from pytorch_lightning.loggers import TestTubeLogger
    logger = TestTubeLogger(save_dir=mt.RUNS_DIR, name=args.name,
                            version=args.version)
    exp_dir = Path(logger.save_dir) / logger.name / f"version_{logger.experiment.version}"

    # CALLBACKS
    callbacks = []

    # log learning rate
    from pytorch_lightning.callbacks import LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # checkpointing
    # After training finishes, use :attr:`best_model_path` to retrieve the path to the
    # best checkpoint file and : attr: `best_model_score` to retrieve its score.
    from pytorch_lightning.callbacks import ModelCheckpoint
    ckpt_callback = ModelCheckpoint(dirpath=exp_dir / 'checkpoints', filename=None,
                                    monitor='loss/val', save_top_k=3, mode='min')
    callbacks.append(ckpt_callback)

    if ckpt_callback.best_model_path == '':
        best_model_path = None
    else:
        best_model_path = ckpt_callback.best_model_path

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        precision=16,
        # auto_lr_find=True,
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=logger,
        terminate_on_nan=True,
        resume_from_checkpoint=best_model_path,
        # weights_summary='full',
        log_gpu_memory=True,
        gpus=None,#'0' if torch.cuda.is_available() else None,
        profiler=pl.profiler.SimpleProfiler(
            output_filename=exp_dir / 'profiler-report.txt'),
        gradient_clip_val=GRAD_CLIP,
        deterministic=True,
        num_sanity_val_steps=0,
        move_metrics_to_cpu=True)

    # Train
    trainer.fit(model, datamodule=datamodule)

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

    # add datamodule arguments
    parser = mt.data.MetaDataModule.add_argparse_args(parser)

    # add model arguments
    parser = mt.model.ProtoNet.add_model_specific_args(parser)

    args = parser.parse_args()

    train(args)
