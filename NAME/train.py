"""train.py - model training"""


import argparse
from pathlib import Path

import pytorch_lightning as pl

import NAME


###############################################################################
# Train
###############################################################################


def main():
    """Train a model"""
    # Parse command-line arguments
    args = parse_args()

    # Setup tensorboard
    logger = pl.loggers.TensorBoardLogger('logs', name=Path().parent.name)

    # Setup data
    datamodule = NAME.DataModule(args.dataset,
                                 args.batch_size,
                                 args.num_workers)

    # Setup trainer
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    # Train
    trainer.fit(NAME.Model(args), datamodule=datamodule)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(add_help=False)

    # Add project arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='The size of a batch')
    # TODO - If you have one dataset, change default to the name of that
    #        dataset. Otherwise, delete the default.
    parser.add_argument(
        '--dataset',
        default='DATASET',
        help='The name of the dataset')
    parser.add_argument(
        '--num_workers',
        type=int,
        help='Number data loading jobs to launch. If None, uses number of ' +
             'cpu cores.')

    # Add model arguments
    parser = NAME.Model.add_model_specific_args(parser)

    # Add trainer arguments
    parser = pl.Trainer.add_argparse_args(parser)

    # Parse
    return parser.parse_args()


if __name__ == '__main__':
    main()
