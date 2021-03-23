"""partition.py - dataset partitioning"""
import music_trees as mt
from music_trees.tree import MusicTree

from typing import List
from collections import OrderedDict
import argparse
import json
import logging
import random
from pathlib import Path
import copy

from colorama import Fore
from colorama import Style
import medleydb as mdb

DEPTH = 3
TEST_SIZE = 0.3
# these classes do not fit nicely in our hierarchy, either
# because they're too general (horn section) or not an instrument (sampler)
UNWANTED_CLASSES =  ('Main System', 'fx/processed sound', 'sampler', 'horn section', 
                     'string section', 'brass section', 'castanet', 'electronic organ', 'scratches', 'theremin', )

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
if not (mt.ASSETS_DIR / 'mdb-files.json').exists():
    FILES = {inst: get_files_for_instrument(inst) for inst in load_unique_instrument_list()}
    mt.utils.data.save_entry(FILES, mt.ASSETS_DIR / 'mdb-files.json')
else:
    FILES = mt.utils.data.load_entry(mt.ASSETS_DIR / 'mdb-files.json')

def add_track_information_to_tree(tree):
    for node in tree.expand_tree():
        node = tree[node]
        if tree.is_leaf(node):
            node.data['n_tracks'] = len(FILES[node.uid])
        else:
            node.data['n_tracks'] = sum([len(FILES[n.uid]) for n in tree.leaves(node.uid)])

def _flat_split(leaves: List[mt.tree.MusicNode], test_size: float):
    """ split a list of instruments according to given test size"""
    
    leaves = sorted(leaves, key=lambda node: node.data['n_tracks'])

    # split the list of leaf nodes according to the test size
    split_idx = int(round(test_size*len(leaves)))
    
    partition = {}
    partition['train'] = [l.uid for l in leaves[split_idx:]]
    partition['test'] = [l.uid for l in leaves[:split_idx]]
    
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

def _hierarchical_split(tree: MusicTree, depth: int, test_size=TEST_SIZE):
    # get all nodes at said depth
    nodes = tree.all_nodes_at_depth(depth)
    partition = {'train': [], 'test': []}

    for node in nodes:
        leaves = [n for n in tree.leaves(node.uid)]
        local_partition = _flat_split(leaves, test_size=test_size)
        partition = _update_partition(partition, local_partition)

    return partition

def _prune_tree_to_available_instruments(tree):
    available = load_unique_instrument_list()
    all_tags = [n for n in tree.all_nodes()]
    not_available = [n.uid for n in all_tags if n.uid not in available and tree.is_leaf(n)]
    tree.remove_by_tags(not_available)

def add_partition_data_to_tree(tree, partition):
    for split, insts in partition.items():
        for inst in insts:
            node = tree.get_node(inst)
            node.data.update({'partition': split, 'n_tracks': len(FILES[inst])})

def make_partition_subtree(tree: MusicTree, partition: dict, split: str):
    # make a new tree to keep the other one const
    tree = MusicTree(tree=tree, deep=True)

    def in_partition(t, n):
        return (n.uid in partition[split] and t.is_leaf(n)) or not t.is_leaf(n)

    partition_tree = tree.filter_tree(in_partition)
    add_track_information_to_tree(partition_tree)

    return partition_tree

def hierarchical_split():
    """ write me
    should make a 70/30 split
    at the leaves of a tree
    """
    taxonomy = mdb.INST_TAXONOMY
    tree = MusicTree.from_taxonomy(taxonomy)

    # shorten tree to desired depth
    tree = tree.shorten(DEPTH)

    # prune out unwanted classes
    # remove these two categories we know we don't want
    tree.remove_by_tags(['other', 'electric'])
    _prune_tree_to_available_instruments(tree)

    add_track_information_to_tree(tree)

    tree = tree.even_depth()
    partition = _hierarchical_split(tree, depth=DEPTH, test_size=TEST_SIZE)
    
    print(Fore.MAGENTA)
    print('TRAIN')
    train_tree = make_partition_subtree(tree, partition, 'train')
    train_tree.show(data_property='n_tracks')
    print(Style.RESET_ALL)

    print(Fore.CYAN)
    print('TEST')
    test_tree = make_partition_subtree(tree, partition, 'test')
    test_tree.show(data_property='n_tracks')
    print(Style.RESET_ALL)
    
    logging.info(f'number of training classes: {len(partition["train"])}')
    logging.info(f'number of test classes: {len(partition["test"])}')

    # save the partitions
    save_path = mt.ASSETS_DIR / 'partition.json'
    mt.utils.data.save_entry(partition, save_path, format='json')
    logging.info(f'saved partition map to {save_path}')

if __name__ == "__main__":
    hierarchical_split()
