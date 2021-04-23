# parse copy-pasted wikipedia text for hornbostel
# to build a hierarchy.

import music_trees as mt
from music_trees.tree import MusicTree

import re

src = mt.ASSETS_DIR / 'taxonomies' / 'wiki-hornbostel.txt'
dest = mt.ASSETS_DIR / 'taxonomies' / 'hornbostel-mdb.yaml'

src_tree = MusicTree.from_taxonomy(mt.utils.data.load_entry(mt.ASSETS_DIR / 'taxonomies' /
                                                            'deep-mdb.yaml', format='yaml'))
all_leaves = src_tree.leaves(src_tree.root)

NODE = '\d+\.?\d*'


def find_node(line):
    out = re.search(NODE, line)
    if out is not None:
        out = out.group(0)
    return out


def clean(string):
    return string.replace('\n', '')


def add_node(tree: MusicTree, node, contents):
    assert isinstance(node, str)

    # add nodes to the tree iteratively
    for idx, char in enumerate(node):
        subnode = node[:idx+1]
        if subnode[-1] == '.':
            continue

        parent = node[:idx] if idx > 0 else tree.root
        if parent[-1] == '.':
            parent = parent[:-1]

        # breakpoint()
        contents = contents.replace(subnode, '').lower()
        contents = clean(contents)
        if not tree.contains(subnode):
            tree.create_node(subnode, parent=parent)

        nd = tree.get_node(subnode)

        if not hasattr(nd.data, 'content') or nd.data.content is None:
            nd.data.content = []

        if contents not in all_leaves:
            continue

        nd.data.content.append(contents)


with open(src) as f:
    lines = f.readlines()

    tree = MusicTree()
    current_node = tree.root

    for i, line in enumerate(lines):
        new_node = find_node(line)
        current_node = new_node if new_node is not None else current_node
        if current_node is not tree.root:
            add_node(tree, current_node, line)

    breakpoint()
