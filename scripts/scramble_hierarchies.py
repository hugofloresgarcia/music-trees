import music_trees as mt
from music_trees.tree import MusicTree

from copy import deepcopy
import random

from tqdm import tqdm

NUM_TAXONOMIES = 10
NUM_SHUFFLES = 1000

output_dir = mt.ASSETS_DIR / 'taxonomies'
output_dir.mkdir(exist_ok=True)

target_tree = mt.utils.data.load_entry(
    mt.ASSETS_DIR / 'taxonomies' / 'deeper-mdb.yaml', format='yaml')
target_tree = MusicTree.from_taxonomy(target_tree)


def scramble_tree(tree: MusicTree):
    "scramble a class tree"

    # first, copy the tree
    tree = deepcopy(tree)

    # shuffle many times
    for _ in tqdm(list(range(NUM_SHUFFLES))):

        # get all of the leaves twice
        A = tree.leaves()
        B = tree.leaves()

        # scramble one of them
        random.shuffle(B)

        # swap a and b for all A and B
        for an, bn in zip(A, B):
            tree.swap_leaves(an, bn)

    return tree


def export_tree(tree: MusicTree, fp: str):
    mt.utils.data.save_entry(tree._2dict(), fp, format='yaml')


if __name__ == "__main__":

    for i in range(NUM_TAXONOMIES):
        t = scramble_tree(target_tree)
        # breakpoint()
        fp = output_dir / f'scrambled-{i}'
        export_tree(t, fp)
