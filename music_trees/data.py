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

class MDBMeta(torch.utils.data.Dataset):

    def __init__(self, partition: str, n_class: int, n_shot: int,
                n_query: int, duration: float = 0.5, 
                sample_rate: int = 16000, transform=None):
        super().__init__()
        # load the classlist for this partition
        self.classes =  mt.utils.data.load_entry( mt.ASSETS_DIR / 'partition.json', 
                                                          format='json')[partition]
        self.files = self._load_files()

        self.n_class = n_class
        self.n_shot = n_shot
        self.n_query = n_query

        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration
    
    def _load_files(self):
        files = {}
        for name in self.classes:
            files[name] = mdb.get_files_for_instrument(name)

    def _get_example_for_class(self, name):
        # grab a random file
        file = random.choice(self.files[name])

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
        #NOTE: __getitem__ should return a single item
        # and get_episode should return an episode?
        return

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
        """Retrieve the PyTorch DataLoader for testing"""
        return loader(self.name, 'test', self.batch_size, self.num_workers)

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