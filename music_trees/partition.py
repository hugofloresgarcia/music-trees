"""partition.py - dataset partitioning"""
import music_trees as mt

from typing import List
from collections import OrderedDict
import argparse
import json
import logging
import random
from pathlib import Path

from colorama import Fore
from colorama import Style
import medleydb as mdb

TEST_SIZE = 0.3
# these classes do not fit nicely in our hierarchy, either
# because they're too general (horn section) or not an instrument (sampler)
UNWANTED_CLASSES =  ('Main System', 'fx/processed sound', 'sampler', 'horn section',
                     'string section', 'brass section', 'castanet', 'electronic organ', 'scratches')

# because instrument sections are the same as the instrument, we want them to be considered 
# as one instrument
REMAP_CLASSES = {'violin section': 'violin', 
                 'viola section': 'viola',
                 'french horn section': 'french horn',
                 'trombone section': 'trombone',
                 'flute section': 'flute', 
                 'clarinet section': 'clarinet', 
                 'cello section': 'cello'}

def load_unique_instrument_list():
    # the first thing to do is to partition the MDB track IDs and stem IDs into only the ones we will use.
    mtracks = mdb.load_all_multitracks(['V1', 'V2'])

    # gather an instrument list
    instruments = []
    for mtrack in mtracks:
        instruments.extend([stem.instrument[0] for stem in mtrack.stems.values() if stem.audio_path is not None])

    instruments = list(set(instruments))

    # filter out classes
    instruments = [i for i in instruments if i not in UNWANTED_CLASSES and i not in REMAP_CLASSES]
    logging.info(f'classlist is: {instruments}')

    return instruments

def get_files_for_instrument(instrument: str):
    files = list(mdb.get_files_for_instrument(instrument))
    # find out if we have any other files from the remapped section
    for key, val in REMAP_CLASSES.items():
        if instrument == val:
            files.extend(list(mdb.get_files_for_instrument(key)))
    return files

# keep track of each instrument and its list of files
FILES = {inst: get_files_for_instrument(inst) for inst in load_unique_instrument_list()}

def _flat_split(instruments: List[str], test_size: float):
    """ split a list of instruments according to given test size"""

    # get a counted subset of the instruments
    instruments = {inst: len(FILES[inst]) for inst in instruments}
    instruments = list(OrderedDict(sorted(instruments.items(), key=lambda x: x[1])).keys())

    # split the list of leaf nodes according to the test size
    split_idx = int(round(test_size*len(instruments)))
    
    partition = {}
    partition['train'] = instruments[split_idx:]
    partition['test'] = instruments[:split_idx]
    
    return partition

def _update_partition(this: dict, other: dict):
    """ given two dicts of lists (this and other), 
    extends the list of `this` with the contents of `other`
    NOTE: they must have exactly the same keys or will raise an assertion error
    NOTE: not done in place (returns a copy of the dict)
    """
    this = dict(this)
    for key in this:
        assert key in other
        assert isinstance(this[key], list)
        assert isinstance(other[key], list)

        this[key].extend(other[key])
    return this

def _hierarchical_split(tree: dict, partition: dict, test_size: float):
    for name, node in tree.items():
        if isinstance(node, dict):
            # recurse!
            local_partition = _hierarchical_split(node, partition, test_size)
            partition = _update_partition(partition, local_partition)

        elif isinstance(node, list):
            # only get the valid instruments
            breakpoint()
            node = [inst for inst in node if inst in FILES.keys()]
            # make the split and update the partitions
            local_partition = _flat_split(node, test_size)
            partition = _update_partition(partition, local_partition)

    return partition

def _pretty_print_hierarchical_split(tree: dict, partition: dict, level: int):
    for name, node in tree.items():
        if isinstance(node, dict):
            print(level * '\t' + f"{name}")
            _pretty_print_hierarchical_split(node, partition, level+1)
        elif isinstance(node, list):
            colors = [Fore.CYAN, Fore.GREEN]
            local_partition = {}
            print(level * '\t' + f"{name}")

            for (split, instruments), color in zip(partition.items(), colors): 
                local_partition[split] = [i for i in instruments if i in node]
                print(color, end='')
                print((level+1) * '\t' + f"{local_partition[split]}")
                print(Style.RESET_ALL, end='')

def hierarchical_split():
    """ write me
    should make a 70/30 split
    at the leaves of a tree
    """
    # we will be using a new taxonomy
    # taxonomy = mt.utils.data.load_entry(
    #     mt.ASSETS_DIR / 'base-taxonomy.yaml', format='yaml')
    taxonomy = mdb.INST_TAXONOMY

    partition = {'train': [], 'test': []}
    partition = _hierarchical_split(taxonomy, partition, TEST_SIZE)

    _pretty_print_hierarchical_split(taxonomy, partition, 0)

    logging.info(f'number of training classes: {len(partition["train"])}')
    logging.info(f'number of test classes: {len(partition["test"])}')

    # save the partitions
    save_path = mt.ASSETS_DIR / 'partition.json'
    mt.utils.data.save_entry(partition, save_path, format='json')
    logging.info(f'saved partition map to {save_path}')

if __name__ == "__main__":
    hierarchical_split()
