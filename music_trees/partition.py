"""partition.py - dataset partitioning"""
import music_trees as mt
from music_trees.tree import MusicTree

from collections import OrderedDict
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
            node.data['records'] = mt.utils.data.filter_records_by_class_subset(records, [
                                                                                node.uid])
        else:
            node.data['n_examples'] = sum(
                [freqs[n.uid] for n in tree.leaves(node.uid)])


def percentage_split(seq, percentages):
    """https://stackoverflow.com/questions/14280856/separate-a-list-into-four-parts-based-on-percentage-even-if-the-list-is-not-divi"""
    assert sum(percentages) == 1.0
    prv = 0
    size = len(seq)
    cum_percentage = 0
    for p in percentages:
        cum_percentage += p
        nxt = int(cum_percentage * size)
        yield seq[prv:nxt]
        prv = nxt


def _flat_split(leaves: List[mt.tree.MusicNode], partitions, sizes):
    """ split a list of instruments according to given test size"""
    leaves = sorted(
        leaves, key=lambda node: node.data['n_examples'], reverse=True)

    partition = {}
    for key, split in zip(partitions, percentage_split(leaves, sizes)):
        partition[key] = {l.uid: l.data['records'] for l in split}

    return partition


def _update_partition(this: dict, other: dict):
    """ given two dicts of dicts (this and other),
    extends the dict of `this` with the contents of `other`
    NOTE: they must have exactly the same keys or will raise an assertion error
    NOTE: not done in place (returns a copy of the dict)
    """
    this = dict(this)
    for key in this:
        assert key in other
        assert isinstance(this[key], dict)
        assert isinstance(other[key], dict)

        this[key].update(other[key])
    return this


def _hierarchical_split(tree: MusicTree, depth: int, partitions, sizes):
    # get all nodes at said depth
    nodes = tree.all_nodes_at_depth(depth)
    partition = OrderedDict((p, {}) for p in partitions)

    for node in nodes:
        leaves = [n for n in tree.leaves(node.uid)]
        local_partition = _flat_split(leaves, partitions, sizes)
        partition = _update_partition(partition, local_partition)

    return partition


def make_partition_subtree(tree: MusicTree, partition: dict, split: str, records: dict):
    # make a new tree to keep the other one const
    tree = MusicTree(tree=tree, deep=True)
    partition_tree = tree.fit_to_classlist(partition[split])
    populate_tree_with_class_statistics(partition_tree, records)

    return partition_tree


def display_partition_subtrees(tree, partition, records):
    colors = [Fore.MAGENTA, Fore.CYAN, Fore.GREEN, Fore.BLUE]

    for name in partition:
        print(colors.pop(-1))
        print(name.upper())
        ptree = make_partition_subtree(tree, partition, name, records)
        ptree.show(data_property='n_examples')
        print(Style.RESET_ALL)

        logging.info(f'number of {name} classes: {len(partition[name])}')


def hierarchical_partition(taxonomy: str, name: str, partitions: List[str],
                           sizes: List[float], depth: int):
    taxonomy = mt.utils.data.load_entry(
        mt.ASSETS_DIR / 'taxonomies' / f'{taxonomy}.yaml', format='yaml')
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

    # we want to remove all classes not present in the desired taxonomy
    all_leaves = [n.uid for n in tree.leaves(tree.root)]
    classlist = [c for c in classlist if c in all_leaves]
    tree.fit_to_classlist(classlist)
    populate_tree_with_class_statistics(tree, records)

    assert len(partitions) == len(sizes)
    partition = _hierarchical_split(tree, depth, partitions, sizes)

    display_partition_subtrees(tree, partition, records)

    # save the partitions
    save_path = mt.ASSETS_DIR / name / 'partition.json'
    mt.utils.data.save_entry(partition, save_path, format='json')
    logging.info(f'saved partition map to {save_path}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--taxonomy', type=str, required=True,
                        help='taxonomy to use for dataset split')

    parser.add_argument('--name', type=str, required=True,
                        help='name of the dataset to partition. must be a subdir\
              in core.DATA_DIR')

    parser.add_argument('--partitions', type=str, required=True, nargs='+',
                        help='name of partitions to create, example: "--partitions train val test"')

    parser.add_argument('--sizes', type=float, required=True, nargs='+',
                        help='proportion of partitions to allocate to each partition, example: "--sizes 0.3 0.5 0.2"')

    parser.add_argument('--depth',     type=int, default=2,
                        help='depth of the tree on which to perform the split')

    hierarchical_partition(**vars(parser.parse_args()))
