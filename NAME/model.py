"""model.py - model definition"""


import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn


###############################################################################
# Model
###############################################################################


class Model(pl.LightningModule):
    """PyTorch Lightning model definition"""

    # TODO - add hyperparameters as input args
    def __init__(self):
        super().__init__()

        # Save hyperparameters with checkpoints
        self.save_hyperparameters()

        # TODO - define model
        raise NotImplementedError

    ###########################################################################
    # Forward pass
    ###########################################################################

    def forward(self):
        """Perform model inference"""
        # TODO - define model arguments and implement forward pass
        raise NotImplementedError

    ###########################################################################
    # PyTorch Lightning - model-specific argparse argument hook
    ###########################################################################

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model hyperparameters as argparse arguments"""
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # TODO - add hyperparameters as command-line args using
        #        parser.add_argument()
        return parser

    ###########################################################################
    # PyTorch Lightning - step model hooks
    ###########################################################################

    def training_step(self, batch, index):
        """Performs one step of training"""
        # TODO - implement training step
        raise NotImplementedError

    def validation_step(self, batch, index):
        """Performs one step of validation"""
        # TODO - implement validation step
        raise NotImplementedError

    def test_step(self, batch, index):
        """Performs one step of testing"""
        # OPTIONAL - only implement if you have meaningful objective metrics
        raise NotImplementedError

    ###########################################################################
    # PyTorch Lightning - optimizer
    ###########################################################################

    def configure_optimizer(self):
        """Configure optimizer for training"""
        return torch.optim.Adam(self.parameters())
