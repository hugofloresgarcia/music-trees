"""data.py - data loading"""


import os
import random
import logging
from pathlib import Path
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import numpy as np
from nussl import AudioSignal
import music_trees as mt
import shelve
import tqdm

import unicodedata
import re

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

class MetaDataset(torch.utils.data.Dataset):

    def __init__(self, name: str, partition: str, n_class: int, n_shot: int,
                n_query: int, audio_tfm=None, epi_tfm=None, clear_cache=False):
        """pytorch dataset for meta learning. 

        Args:
            name (str): name of the dataset. Must be a dirname under core.DATA_DIR
            partition (str): partition. One of 'train', or 'test', or 'validation', as defined in core.ASSETS_DIR / name / 'partition.json'
            n_class (int): number of classes in episode (n_way)
            n_shot (int): number of support examples per class
            n_query (int): number of query example
            transform ([type], optional): [description]. transform to apply. If none, returns an AudioSignal
        """
        super().__init__()
        # load the classlist for this partition
        self.root = mt.DATA_DIR / name 
        self.classes =  mt.utils.data.load_entry(mt.ASSETS_DIR / name / 'partition.json', 
                                                          format='json')[partition]
        self.classes.sort()
        self.files = self._load_files()

        self.n_class = n_class
        self.n_shot = n_shot
        self.n_query = n_query

        self.audio_tfm = audio_tfm
        self.epi_tfm = epi_tfm

        self.shelf_path = Path(mt.CACHE_DIR / name / partition / 'cache').with_suffix('.pag')
        self.shelf_path.parent.mkdir(exist_ok=True, parents=True)
        if self.shelf_path.exists() and clear_cache:
            logging.warn(f'clear_cache = True and cache exists. clearing {self.shelf_path}')
            os.remove(self.shelf_path)

        self.cache_dataset()

        self._debug_episode = None
        self.debug = True

    def __len__(self):
        """ the sum of all available clips, times the number of classes per episode / the number of shots"""
        return sum([len(entries) for entries in self.files.values()])

    def _load_files(self):
        files = {}

        for name in self.classes:
            records = mt.utils.data.glob_all_metadata_entries(self.root / name)
            files[name] = records
        
        # sort by key
        files = OrderedDict(sorted(files.items(), key=lambda x: x[0]))

        assert list(files.keys()) == self.classes,\
             f"classlist-data mismatch {files.keys()}, {self.classes}"
            
        return files

    def cache_dataset(self, chunksize=1000):
        logging.info(f'caching dataset...')
        with shelve.open(str(self.shelf_path), writeback=True) as shelf:
            # go through all classnames
            for cl, records in self.files.items():
                for i, entry in tqdm.tqdm(list(enumerate(records))):
                # all files belonging to a class
                    self.cache_if_needed(shelf, entry)
                    if i % chunksize:
                        shelf.sync()

    def cache_if_needed(self, shelf, entry: dict):
        if entry['uuid'] in shelf:
            cached_entry = shelf[entry['uuid']]
        else:
            cached_entry = self.transform_entry(entry)
            shelf[entry['uuid']] = cached_entry
        return cached_entry

    def transform_entry(self, entry: dict):
        # load audio
        audio_path = Path(mt.utils.data.get_path(entry)).with_suffix('.wav')
        signal = AudioSignal(path_to_input_file=str(
            audio_path)).to_mono(keep_dims=True)

        entry['audio'] = signal

        if self.audio_tfm is not None:
            entry = self.audio_tfm(entry)
        return entry

    def process_entry(self, entry: dict):
        # no need to open the shelf in this case
        if self.audio_tfm is None:
            return entry

        with shelve.open(str(self.shelf_path), writeback=False) as shelf:
            cached_entry = self.cache_if_needed(shelf, entry)
        return cached_entry

    def _get_example_for_class(self, name: str):
        # grab a random file
        entry = dict(random.choice(self.files[name]))
        entry = self.process_entry(entry)

        return entry

    def __getitem__(self, index: int):
        """returns a dict with format:

        episode = {
            'support': List[AudioSignal], 
            'query': List[AudioSignal], 
            'classes': List[str],
        }
        """
        if self.debug:
            if self._debug_episode is not None:
                return self._debug_episode
            else:
                subset = random.sample(self.classes, k=self.n_class)
                subset.sort()

                # grab the support set
                support = OrderedDict([
                    (name, [self._get_example_for_class(name)
                            for _ in range(self.n_shot)])
                    for name in subset
                ])
                
                # grab the query set
                # grab n_query examples per class
                query = []
                for pick in subset:
                    for _ in range(self.n_query):
                        query.append(self._get_example_for_class(pick))

                episode = {
                    'support': support, 
                    'query': query, 
                    'classes': subset
                }

                if self.epi_tfm is not None:
                    episode = self.epi_tfm(episode)
                self._debug_episode = episode                    
        return episode

class MetaDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module

    Arguments
        name - string
            The name of the dataset
        batch_size - int
            The size of a batch
        num_workers - int or None
            Number data loading jobs to launch. If None, uses num cpu cores.
        **kwargs: 
            Any kwargs for MetaDataset. 
    """

    def __init__(self, name, batch_size=64, num_workers=None, **kwargs):
        super().__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.kwargs = kwargs
    
    def setup(self, stage):
        # load all partitions
        partition = mt.utils.data.load_entry(mt.ASSETS_DIR / self.name / 'partition.json')
        splits = list(partition.keys())

        assert 'train' in partition

        self.dataset = MetaDataset(self.name, partition='train', **self.kwargs)
        
        if 'val' in partition:
            self.val_dataset = MetaDataset(self.name, partition='val', **self.kwargs)
        else:
            self.val_dataset = MetaDataset(
                self.name, partition='test', **self.kwargs)

        if 'test' in partition:
            pass
            #self.test_dataset = MetaDataset(self.name, partition='test', **self.kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, required=False)
        return parser

    def train_dataloader(self):
        """Retrieve the PyTorch DataLoader for training"""
        return loader(self.dataset, 'train', self.batch_size, self.num_workers)

    def val_dataloader(self):
        """Retrieve the PyTorch DataLoader for validation"""
        assert hasattr(self, 'val_dataset')
        return loader(self.val_dataset, 'validation', self.batch_size, self.num_workers)

    def test_dataloader(self):
        """Retrieve the PyTorch DataLoader for testing"""
        assert hasattr(self, 'test__dataset')
        return loader(self.test_dataset, 'test', self.batch_size, self.num_workers)

def episode_collate(batch):
    """ 
    expect batch to be a list of dicts with numpy arrays
    will only collate the keys in 
    """
    # get the keys we expect
    keys = batch[0].keys()
    output = {}

    for key in keys:
        exmpl = batch[0][key]
        if isinstance(exmpl, np.ndarray):
            stack = np.stack([item[key] for item in batch])
            output[key] = torch.from_numpy(stack)
        elif isinstance(exmpl, torch.Tensor):
            output[key] = torch.stack([item[key] for item in batch])
        else:
            output[key] = [item[key] for item in batch]

    return output

def loader(dataset, partition, batch_size=64, num_workers=None):
    """Retrieve a data loader"""
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle='train' in partition,
        num_workers=os.cpu_count() if num_workers is None else num_workers,
        pin_memory=True,
        collate_fn=episode_collate)
