from random import choice

from delab_trees.delab_tree import DelabTree


class DelabTrees:

    def __init__(self, df):
        self.trees = {}
        self.df = df
        self.initialize_trees()

    def initialize_trees(self):
        grouped_by_tree_ids = {k: v for k, v in self.df.groupby("tree_id")}
        for tree_id, df in grouped_by_tree_ids.items():
            tree = DelabTree(df)
            tree.initialize_networkx_tree()
            self.trees[tree_id] = tree

    def single(self):
        assert len(self.trees.keys()) == 1
        return self.trees[self.trees.keys()[0]]

    def one(self):
        assert len(self.trees.keys()) >= 1
        return self.trees[choice(self.trees.keys())]
