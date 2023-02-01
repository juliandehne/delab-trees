import pickle
from random import choice

import pandas as pd

from delab_trees.constants import TREE_IDENTIFIER
from delab_trees.delab_tree import DelabTree
from delab_trees.algorithms.preperation_alg_rb import prepare_rb_data
from delab_trees.algorithms.training_alg_rb import train_rb


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

    def __prepare_rb_model(self, prepared_data_filepath):
        """
        convert the trees to a matrix with all the node pairs (node -> ancestor) in order run the RB
        algorithm that wagers a prediction whether a post has been seen by a subsequent post's author

        """
        rb_prepared_data = prepare_rb_data(self)
        rb_prepared_data.to_pickle(prepared_data_filepath)
        return rb_prepared_data

    def __train_apply_rb_model(self, prepared_data_filepath="rb_prepared_data.pkl", prepare_data=True) \
            -> dict['tree_id', dict['author_id', 'rb_vision']]:
        if prepare_data:
            data = self.__prepare_rb_model(prepared_data_filepath)
        else:
            with open("rb_prepared_data.pkl", 'rb') as f:
                data = pickle.load(f)
        assert data.empty is False, "There was a mistake during the preparation of the data for the rb algorithm"
        applied_model, trained_model, features = train_rb(data)
        result = applied_model.pivot(columns='conversation_id', index='author', values="predictions").to_dict()
        return result

    def get_rb_vision(self, tree: DelabTree = None):
        applied_rb_model = self.__train_apply_rb_model()
        if tree is None:
            return applied_rb_model
        else:
            return applied_rb_model[tree.conversation_id]

    def __prepare_pb_model(self):
        pass

    def train_apply_pb_model(self):
        pass
