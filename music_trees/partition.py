"""partition.py - dataset partitioning"""
import music_trees as mt

from typing import OrderedDict
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
UNWANTED_PARENTS = ('electric', 'other')

# because instrument sections are the same as the instrument, we want them to be considered 
# as one instrument
REMAP_CLASSES = {'violin section': 'violin', 
                 'viola section': 'viola',
                 'french horn section': 'french horn',
                 'trombone section': 'trombone',
                'flute section': 'flute', 
                'clarinet section': 'clarinet'}

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
    # TODO: maybe make this faster? 
    files = list(mdb.get_files_for_instrument(instrument))
    # find out if we have any other files from the remapped section
    for key, val in REMAP_CLASSES.items():
        if instrument == val:
            files.extend(list(mdb.get_files_for_instrument(key)))
    return files

if __name__ == "__main__":
    # get the unique list of instruments
    instruments = load_unique_instrument_list()

    # we will be using a new taxonomy 
    taxonomy = mt.utils.data.load_metadata_entry(mt.ASSETS_DIR / 'base-taxonomy.yaml', format='yaml')

    # our redacted hierarchy
    hierarchy = {}
    partitions = {'train': [], 'test': []}
    for parent, subparent_dict in taxonomy.items():
        if parent in UNWANTED_PARENTS:
            continue
        
        parent_list = []
        for subparent_name, subparent_list in subparent_dict.items():
            # only keep the instruments we previously defined
            subparent_list = [i for i in subparent_list if i in instruments]
            parent_list.extend(subparent_list)

        # sort the parent list by number of examples in descending order
        counts = ([(instrument, len(get_files_for_instrument(instrument))) for instrument in parent_list])
        counts = sorted(counts, key=lambda x: x[1])

        # split the list of leaf nodes according to the test size
        split_idx = int(round(TEST_SIZE*len(counts)))
        
        train_subpartition = counts[split_idx:]
        test_subpartition = counts[:split_idx]

        print(f'PARENT: {Fore.GREEN}{parent}{Style.RESET_ALL}')
        print(f'\t TRAIN: {Fore.BLUE}{train_subpartition}{Style.RESET_ALL}')
        print(f'\t TEST: {Fore.MAGENTA}{test_subpartition}{Style.RESET_ALL}')

        # remove the counts for the actual partition
        train_subpartition = [c[0] for c in train_subpartition]
        test_subpartition = [c[0] for c in test_subpartition]

        partitions['train'].extend(train_subpartition)
        partitions['test'].extend(test_subpartition)

        for instrument in parent_list:
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(instrument)

    logging.info(f'number of training classes: {len(partitions["train"])}')
    logging.info(f'number of test classes: {len(partitions["test"])}')

    # save the partitions
    save_path = mt.ASSETS_DIR / 'partition.json'
    mt.utils.data.save_metadata_entry(partitions, save_path, format='json')
    logging.info(f'saved partition map to {save_path}')

    # save the hierarchy 
    save_path = mt.ASSETS_DIR / 'hierarchy.json'
    mt.utils.data.save_metadata_entry(hierarchy, save_path, format='yaml')
    logging.info(f'saved class hierarchy to {save_path}')