""" tree.py - TODO: write desc and fill docstrings """
from copy import deepcopy
import warnings

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

    @staticmethod
    def new_unique(other):
        """ 
        creates a new unique
        instance, copying the data but adding a _ to its id
        the pointers are not copied. 
        """
        new = MusicNode(uid=other.uid+'_')
        new.data = deepcopy(other.data)
        return new

    @property
    def uid(self):
        return self.identifier

    def __repr__(self):
        return f"{self.uid}-{self.data.keys()}"

    # def __deepcopy__(self, memo=None):
    #     # make a deepcopy of self, but give it a new id
    #     other = deepcopy(self, memo=memo)


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
        return super().depth(node) + 1

    def create_node(self, uid: str, parent=None):
        return super().create_node(identifier=uid, parent=parent)

    def _r_make_tree(self, parent: Node, taxonomy: dict):
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
        this = MusicTree(node_class=MusicNode)
        root = this.root
        this._r_make_tree(root, taxonomy)
        return this

    def fit_to_classlist(self, classlist: list):
        """ 
        prunes all leaves and from the tree so the only
        remaining leaves are those in classlist. 

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
            if get_shallow_leaves is True, also returns 
            all leaf nodes shallower than depth
        """
        nodes = []
        for node in self.all_nodes():
            if self.depth(node) == depth:
                nodes.append(node)
        return nodes

    def remove_by_tags(self, tags: list):
        for tag in tags:
            node = self.get_node(tag)
            if node is not None:
                self.remove_node(node.uid)
        return self

    def is_leaf(self, node):
        return self.is_branch(node.uid) == []

    def filter_tree(self, condition: callable):
        """
        creates a new subtree only for the nodes 
        that satisfy the condition provided by the user

        the input `fn` will be passed the tree and the node 
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
        this is probably unnecessarily O(n^2)

        modifies the tree, such that
        the length of all paths from the root node to any 
        leaf node is equal. This is done by inserting a new
        parent to the leaf node in each path, in a loop until
        the uniform depth requirement is met 
        """
        max_depth = self.depth()

        def expand_path_if_needed(path):
            if len(path) < max_depth:
                # get the leaf's parent
                parent = self[path[-2]]
                leaf = self[path[-1]]
                # create a copy of it
                new_parent = MusicNode.new_unique(parent)

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
        all_paths = self.paths_to_leaves()
        pred_path = list(reversed([p for p in all_paths if p[-1] == pred][0]))
        truth_path = list(
            reversed([p for p in all_paths if p[-1] == truth][0]))

        assert len(pred_path) == len(truth_path)
        for h, (pred_ancestor, truth_ancestor) in enumerate(zip(pred_path, truth_path)):
            if pred_ancestor == truth_ancestor:
                return h

        raise ValueError

    def get_ancestor(self, nid: str, height: int):
        """ finds the ancestor of the node at a particular height. 
        if the height is 0, returns the node itself
        """
        all_paths = self.paths_to_leaves()
        node_path = list(reversed([p for p in all_paths if p[-1] == nid]))
        assert len(node_path) == 1
        node_path = node_path[0]

        assert height < len(node_path)

        return node_path[height]


if __name__ == "__main__":
    taxonomy = mdb.INST_TAXONOMY

    tree = MusicTree.from_taxonomy(taxonomy)
    print(tree.hlca('xylophone', 'marimba'))  # 0
    print(tree.hlca('violin', 'viola'))  # 0
    print(tree.hlca('dulcimer', 'sitar'))  # 1
    print(tree.hlca('dizi', 'sitar'))  # 2
