"""partition.py - dataset partitioning"""
import music_trees as mt
from music_trees.tree import MusicTree

from typing import List
import argparse
import logging

from colorama import Fore
from colorama import Style
import medleydb as mdb

def populate_tree_with_class_statistics(tree, records):
    freqs = mt.utils.data.get_class_frequencies(records)

    for node in tree.expand_tree():
        node = tree[node]
        if tree.is_leaf(node):
            node.data['n_examples'] = freqs[node.uid]
        else:
            node.data['n_examples'] = sum([freqs[n.uid] for n in tree.leaves(node.uid)])

def _flat_split(leaves: List[mt.tree.MusicNode], test_size: float):
    """ split a list of instruments according to given test size"""
    
    leaves = sorted(leaves, key=lambda node: node.data['n_examples'])

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

def _hierarchical_split(tree: MusicTree, depth: int, test_size):
    # get all nodes at said depth
    nodes = tree.all_nodes_at_depth(depth)
    partition = {'train': [], 'test': []}

    for node in nodes:
        leaves = [n for n in tree.leaves(node.uid)]
        local_partition = _flat_split(leaves, test_size=test_size)
        partition = _update_partition(partition, local_partition)

    return partition

def make_partition_subtree(tree: MusicTree, partition: dict, split: str, records: dict):
    # make a new tree to keep the other one const
    tree = MusicTree(tree=tree, deep=True)
    partition_tree = tree.fit_to_classlist(partition[split])
    populate_tree_with_class_statistics(partition_tree, records)

    return partition_tree

def hierarchical_partition(name: str, test_size: float, depth: int):
    taxonomy = mdb.INST_TAXONOMY
    tree = MusicTree.from_taxonomy(taxonomy)

    # shorten tree to desired depth
    tree = tree.shorten(depth)

    # make sure the tree is even
    tree = tree.even_depth()

    # prune out unwanted classes
    # remove these two categories we know we don't want
    tree.remove_by_tags(['other'])

    # load the records and get the classlist
    records = mt.utils.data.glob_all_metadata_entries(mt.DATA_DIR / name)
    classlist = mt.utils.data.get_classlist(records)

    tree.fit_to_classlist(classlist)
    populate_tree_with_class_statistics(tree, records)

    partition = _hierarchical_split(tree, depth, test_size)

    print(Fore.MAGENTA)
    print('TRAIN')
    train_tree = make_partition_subtree(tree, partition, 'train', records)
    train_tree.show(data_property='n_examples')
    print(Style.RESET_ALL)

    print(Fore.CYAN)
    print('TEST')
    test_tree = make_partition_subtree(tree, partition, 'test', records)
    test_tree.show(data_property='n_examples')
    print(Style.RESET_ALL)

    logging.info(f'number of training classes: {len(partition["train"])}')
    logging.info(f'number of test classes: {len(partition["test"])}')

    # save the partitions
    save_path = mt.ASSETS_DIR / name / 'partition.json'
    mt.utils.data.save_entry(partition, save_path, format='json')
    logging.info(f'saved partition map to {save_path}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True,
        help='name of the dataset to partition. must be a subdir\
              in core.DATA_DIR')

    parser.add_argument('--test_size', type=float, default=0.3, 
        help='test size')
        
    parser.add_argument('--depth',     type=int, default=2, 
        help='depth of the tree on which to perform the split')
    
    hierarchical_partition(**vars(parser.parse_args()))
