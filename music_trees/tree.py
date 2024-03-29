from copy import deepcopy

import medleydb as mdb
from treelib.tree import Tree
from treelib.node import Node


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDict(deepcopy(dict(self), memo=memo))


class MusicNode(Node):

    def __init__(self, uid: str = None, **kwargs):
        if uid is not None:
            kwargs['identifier'] = uid
        super().__init__(**kwargs)
        if self.data is None:
            self.data = DotDict()

    @property
    def uid(self):
        return self.identifier

    def __repr__(self):
        return f"{self.uid}-{self.data.keys()}"


class MusicTree(Tree):

    def __init__(self, **kwargs):
        kwargs['node_class'] = MusicNode
        super().__init__(**kwargs)

        if not hasattr(self, 'root'):
            self.create_node('root')
        else:
            if self.root is None:
                self.create_node('root')

    def depth(self, node=None):
        """ return the depth of the tree"""
        return super().depth(node)

    def create_node(self, uid: str, parent=None):
        return super().create_node(identifier=uid, parent=parent)

    def _r_make_tree(self, parent: Node, taxonomy: dict):
        """ recursive function for building a tree from a
        taxonomy dict.
        """
        for name, subtaxonomy in taxonomy.items():
            if isinstance(subtaxonomy, dict):
                child = self.create_node(name, parent=parent)
                self._r_make_tree(child, subtaxonomy)
            elif isinstance(subtaxonomy, list):
                child = self.create_node(name, parent=parent)
                leaves = [self.create_node(inst, parent=child)
                          for inst in subtaxonomy]
            else:
                raise ValueError(f"expected either a list of dict: {taxonomy}")

    @staticmethod
    def from_taxonomy(taxonomy: dict):
        """ creates a new tree from an instrument taxonomy dict
        """
        this = MusicTree(node_class=MusicNode)
        root = this.root
        this._r_make_tree(root, taxonomy)
        return this

    def fit_to_classlist(self, classlist: list):
        """
        fits a tree to a classlist.
        that is, given a list of fine-grained labels, this function
        will remove all leaf nodes in this tree whose uids are not
        present in the classlist.

        if a class from `classlist` is not a leaf node in the tree,
        raises an error.
        """
        # first, assert that we will find the whole classlist
        # as leaf nodes in the tree
        leaf_ids = [n.uid for n in self.leaves(self.root)]
        for name in classlist:
            if name not in leaf_ids:
                raise ValueError(
                    f"{name} is missing from the leaves of the tree: {leaf_ids}")

        def _leaves_eq_classlist(classlist):
            leaf_ids = [n.uid for n in self.leaves(self.root)]
            leaf_ids.sort()

            classlist = list(classlist)
            classlist.sort()

            cmpr = [c == l for c, l in zip(classlist, leaf_ids)]
            return all(cmpr)

        while (not _leaves_eq_classlist(classlist)):
            # delete all the leaves whose uids are not in classlist
            leaves = self.leaves(self.root)
            for node in leaves:
                if node.uid not in classlist:
                    self.remove_node(node.uid)

        return self

    def all_nodes_at_depth(self, depth: int):
        """ returns a list of nodes at a given depth
        """
        nodes = []
        for node in self.all_nodes():
            if self.depth(node) == depth:
                nodes.append(node)
        return nodes

    def remove_by_uids(self, uids: list):
        """ remove nodes given a list of node uids """
        for tag in uids:
            node = self.get_node(tag)
            if node is not None:
                self.remove_node(node.uid)
        return self

    def is_leaf(self, node):
        """ is leaf """
        return self.is_branch(node.uid) == []

    def filter_tree(self, condition: callable):
        """
        creates a new subtree only for the nodes
        that satisfy the condition provided by the user

        the arg `condition` will be called like this:
        condition(other, node), where other is a copy of
        this tree, and node is the node being examined at the time.
        """
        other = MusicTree(tree=self, deep=True)
        for node in self.all_nodes():
            # if the condition is not met, then remove this node
            if not condition(other, node):
                other.remove_node(node.uid)
        return other

    def amend_paths(self, amend_fn: callable):
        paths = list(self.paths_to_leaves())
        amends = [True for path in paths]
        while not all([a == False for a in amends]):
            amends = []
            for path in paths:
                amend = amend_fn(path)
                amends.append(amend)
                if amend == True:
                    break
            # because we've changed the paths, we must update them
            paths = list(self.paths_to_leaves())

    def even_depth(self):
        """
        this is probably unnecessarily O(n ^ 2)

        modifies the tree, such that
        the length of all paths from the root node to any
        leaf node is equal. This is done by inserting a new
        parent to the leaf node in each path, in a loop until
        the uniform depth requirement is met
        """
        max_depth = self.depth()+1

        def expand_path_if_needed(path):
            if len(path) < max_depth:
                # get the leaf's parent
                parent = self[path[-2]]
                leaf = self[path[-1]]
                # create a copy of it
                new_parent = MusicNode(parent.uid+'_', data=parent.data)
                # get the children BEFORE we insert our new node
                children = self.children(parent.uid)
                # insert it into the tree as a child to parent
                self.add_node(new_parent, parent=parent)

                # iterate though all the children of parent
                for child in children:
                    # move the leaf as a child to the new parent
                    self.move_node(child.uid, new_parent.uid)

                return True
            else:
                return False

        self.amend_paths(expand_path_if_needed)
        return self

    def shorten(self, depth: int):
        """ remove parents from leaves until
        the maximum depth of the tree is the depth provided
        """
        # we wanna cut the parents so depth is +1
        depth += 1

        def shorten_path_if_needed(path):
            if len(path) > depth:
                # link past the first parent
                parent = path[-2]
                self.link_past_node(parent)
                return True
            else:
                return False

        self.amend_paths(shorten_path_if_needed)
        return self

    def is_even(self):
        """
        returns true if the lengths of all paths from the root
        to each leaf is of equal length
        """
        paths = self.paths_to_leaves()
        len_ = len(paths[0])
        return all([len(path) == len_ for path in paths])

    def hlca(self, pred: str, truth: str):
        """ height of the lowest common ancestor"""
        pred_path = self.get_path_to_root(pred)
        truth_path = self.get_path_to_root(truth)

        assert len(pred_path) == len(truth_path)
        for h, (pred_ancestor, truth_ancestor) in enumerate(zip(pred_path, truth_path)):
            if pred_ancestor == truth_ancestor:
                return h

        raise ValueError

    def hierarchical_precision(self, pred: str, truth: str):
        """ hierachical precision as defined by Kiritchenko"""
        pred_path = set(self.get_path_to_root(pred)[:-1])
        truth_path = set(self.get_path_to_root(truth)[:-1])

        assert len(pred_path) == len(truth_path)

        intersection = len(pred_path & truth_path)
        return intersection / len(pred_path)

    def hierarchical_recall(self, pred: str, truth: str):
        """ hierachical precision as defined by Kiritchenko"""
        pred_path = set(self.get_path_to_root(pred)[:-1])
        truth_path = set(self.get_path_to_root(truth)[:-1])

        assert len(pred_path) == len(truth_path)

        intersection = len(pred_path & truth_path)
        return intersection / len(truth_path)

    def get_path_to_root(self, nid: str):
        path = [nid]

        current_nid = nid
        while current_nid != 'root':
            current_nid = self.parent(current_nid).uid
            path.append(current_nid)

        return path

    def get_ancestor(self, nid: str, height: int):
        """ finds the ancestor of the node at a particular height.
        if the height is 0, returns the node itself
        """
        node_path = self.get_path_to_root(nid)
        assert height < len(node_path)

        return node_path[height]

    def swap_leaves(self, a: MusicNode, b: MusicNode):
        assert self.is_leaf(a) and self.is_leaf(b)

        if a.uid == b.uid:
            return

        # get both parents
        pa = self.parent(a.uid)
        pb = self.parent(b.uid)

        # remove a and b from tree
        self.remove_node(a.uid)
        self.remove_node(b.uid)

        # add a as a child of pb and b as a child of pa
        self.add_node(a, pb)
        self.add_node(b, pa)

        return self

    def _r_2dict(self, node):
        sample_child = self.children(node.uid)[0]
        if self.is_leaf(sample_child):
            return [c.uid for c in self.children(node.uid)]
        else:
            return {c.uid: self._r_2dict(c)
                    for c in self.children(node.uid)}

    def _2dict(self):
        d = {self.root: self._r_2dict(self.get_node(self.root))}
        return d


if __name__ == "__main__":
    taxonomy = mdb.INST_TAXONOMY

    tree = MusicTree.from_taxonomy(taxonomy)
    print(tree.hlca('xylophone', 'marimba'))  # 1
    print(tree.hlca('violin', 'viola'))  # 2
    print(tree.hlca('dulcimer', 'dulcimer'))  # 0
    print(tree.hlca('dizi', 'sitar'))  # 3
