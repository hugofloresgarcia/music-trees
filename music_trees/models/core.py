from music_trees.models.prototree import ProtoTree
import torch
import pytorch_lightning as pl
from music_trees.utils.train import batch_detach_cpu


def load_model(model_name: str):
    raise NotImplementedError
