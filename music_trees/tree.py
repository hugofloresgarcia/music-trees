import medleydb as mdb
from treelib.tree import Tree

class MusicTree:

    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = []
    
    def is_root(self):
        return self.parent is None
    
    def is_leaf(self):
        return self.children == []

    

    def all_leaves(self):
        leaves = []
        for child in self.children:
            if child.is_leaf(): 
                leaves.append(child)
            else: 
                leaves.extend(child.all_leaves())
        return leaves
    
    def __repr__(self):
        if self.is_leaf():
            return f"{self.name}"
        else:
            st = f"{self.name}"
            return st +


def make_tree(tree: Tree, taxonomy: dict):
    for name, subtaxonomy in taxonomy.items():
        if isinstance(subtaxonomy, dict):
            Tree.add_node(name)
            node.make_tree(subtaxonomy)
            self.children.append(node)
        elif isinstance(subtaxonomy, list):
            leaves = [MusicTree(name) for name in subtaxonomy]
            self.children.extend(leaves)
        else:
            raise ValueError(f"expected either a list of dict: {taxonomy}")

if __name__ == "__main__":
    taxonomy = mdb.INST_TAXONOMY
    
    tree = Tree()
    tree.add_ 
