from random import choice

from delab_trees.delab_tree import DelabTree, TREE_IDENTIFIER


class TreeManager:

    def __init__(self, df):
        self.trees = {}
        self.df = df
        self.initialize_trees()

    def initialize_trees(self):
        grouped_by_tree_ids = {k: v for k, v in self.df.groupby(TREE_IDENTIFIER)}
        for tree_id, df in grouped_by_tree_ids.items():
            tree = DelabTree(df)
            tree.as_reply_graph()
            self.trees[tree_id] = tree
        return self

    def single(self) -> DelabTree:
        assert len(self.trees.keys()) == 1, "There needs to be exactly one tree in the manager!"
        return self.trees[self.trees.keys()[0]]

    def random(self) -> DelabTree:
        assert len(self.trees.keys()) >= 1
        return self.trees[choice(list(self.trees.keys()))]
