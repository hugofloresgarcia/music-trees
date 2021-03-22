"""data.py - data loading"""


import json
import os
import random

import pytorch_lightning as pl
import torch
import numpy as np
from nussl import AudioSignal
import librosa
import music_trees as mt
import medleydb as mdb


###############################################################################
# Dataset
###############################################################################

### NEW PLAN
# MDB Meta sends out a dict with two keys:
# support: a dict with structure {classname: AudioSignal}
# query: same thing
# maybe add a to_hierarchical_onehot transform that turns the classname key into a 
# hierarchical onehot.
# BUT it should also include the parent targets

# then, to transforms
# transforms.SomeKindOfAugmentation?
# transforms.STFT and// or Mel if necessary
# transforms.ToEpisode (takes care of stacking queries and making them tensors ready to be input)

class MDBMeta(torch.utils.data.Dataset):

    def __init__(self, partition: str, n_class: int, n_shot: int,
                n_query: int, duration: float = 0.5, 
                sample_rate: int = 16000, transform=None):
        super().__init__()
        # load the classlist for this partition
        self.classes =  mt.utils.data.load_metadata_entry( mt.ASSETS_DIR / 'partition.json', 
                                                          format='json')[partition]

        # load the parent-child hierarchy so we can grab the parents for each instrument
        self.hierarchy = mt.utils.data.load_metadata_entry( mt.ASSETS_DIR / 'hierarchy.yaml', format='yaml')
        self.parents = get_parents_from_hierarchy(self.hierarchy)

        self.n_class = n_class
        self.n_shot = n_shot
        self.n_query = n_query

        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration

    def _get_example_for_class(self, name):
        files = mdb.get_files_for_instrument(name)
        breakpoint()

        # grab a random file
        file = random.choice(files)

        # load audio
        signal = AudioSignal(path_to_input_file=file, sample_rate=self.sample_rate).to_mono()
        
        # remove silence from the audio data
        signal.audio_data = np.expand_dims(mt.utils.effects.trim_silence(signal.audio_data), 0)

        # pick a start point
        start = random.uniform(0, signal.signal_duration - self.duration)
        end = start + self.duration

        # grab that chunk
        signal.audio_data = signal[int(start*signal.sample_rate):int(end*signal.sample_rate)]

        return signal.audio_data

    def __getitem__(self, index):
        # get n_class classes
        class_subset = list(random.sample(self.classes, self.n_class))

        # grab our support set

        # the grab our support set of audio signals
        # the final stacked array should have shape (n_class, n_shot, 1, sample)
        # the labels should have shape (n_class, n_shot)
        support = []
        support_targets = []
        for c in class_subset:
            shots = np.stack([self._get_example_for_class(c) for _ in range(self.n_shot)])
            support.append(shots)

            targets = np.array([self._get_parent_label(c)])
            support_targets.append(targets)
        support = np.stack(support)            
        support_targets = np.stack(support_targets)
        
        # grab the query set of audio signals
        # the stacked array should have shape (n_query, 1, sample)
        # the labels should have shape (n_query)
        query = []
        query_targets = []
        for q in range(self.n_query):
            # pick a random class from the subset to query
            c = random.choice(class_subset)
            query.append(self._get_example_for_class(c))
            query_targets.append(self._get_parent_label(c))
            
        query = np.stack(query)
        query_targets = np.array(query_targets)

        return {
            'query': query,
            'query_targets': query_targets, 
            'support': support, 
            'support_targets': support_targets, 
            'classes': class_subset
            'index': index,
        }

    def __len__(self):
        """Length of the dataset"""
        return len(self.stems)


###############################################################################
# Data module
###############################################################################


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning data module

    Arguments
        name - string
            The name of the dataset
        batch_size - int
            The size of a batch
        num_workers - int or None
            Number data loading jobs to launch. If None, uses num cpu cores.
    """

    def __init__(self, name, batch_size=64, num_workers=None):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        """Retrieve the PyTorch DataLoader for training"""
        # TODO - second argument must be the name of your train partition
        return loader(self.name, 'train', self.batch_size, self.num_workers)

    def val_dataloader(self):
        """Retrieve the PyTorch DataLoader for validation"""
        return loader(self.name, 'valid', self.batch_size, self.num_workers)

    def test_dataloader(self):
        """Retrieve the PyTorch DataLoader for testing"""\
        return loader(self.name, 'test', self.batch_size, self.num_workers)


###############################################################################
# Data loader
###############################################################################

def collate_fn(self, batch):
    raise NotImplementedError

def loader(dataset, partition, batch_size=64, num_workers=None):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle='train' in partition,
        num_workers=os.cpu_count() if num_workers is None else num_workers,
        pin_memory=True,
        collate_fn=collate_fn)

def get_parents_from_hierarchy(hierarchy: dict):
    parents = {}
    for parent, children in hierarchy.items():
        for child in children:
            parents[child] = parent
    return parents
