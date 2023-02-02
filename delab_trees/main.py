import pickle
from random import choice

import pandas as pd

from delab_trees.algorithms.preperation_alg_pb import prepare_pb_data
from delab_trees.algorithms.training_alg_pb import train_pb
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

    def __prepare_pb_model(self, prepared_data_filepath):
        rb_prepared_data = prepare_pb_data(self)
        rb_prepared_data.to_pickle(prepared_data_filepath)
        return rb_prepared_data

    def __train_apply_pb_model(self, prepared_data_filepath="pb_prepared_data.pkl", prepare_data=True) \
            -> dict['tree_id', dict['author_id', 'pb_vision']]:
        if prepare_data:
            data = self.__prepare_pb_model(prepared_data_filepath)
        else:
            with open("pb_prepared_data.pkl", 'rb') as f:
                data = pickle.load(f)
        assert data.empty is False, "There was a mistake during the preparation of the data for the pb algorithm"
        applied_model = train_pb(data)

        result = applied_model.pivot(columns='conversation_id', index='author', values="predictions").to_dict()
        return result

    def get_pb_vision(self, tree: DelabTree = None):
        applied_pb_model = self.__train_apply_pb_model()
        if tree is None:
            return applied_pb_model
        else:
            return applied_pb_model[tree.conversation_id]


def get_test_tree() -> DelabTree:
    from delab_trees import TreeManager

    d = {'tree_id': [1] * 4,
         'post_id': [1, 2, 3, 4],
         'parent_id': [None, 1, 2, 1],
         'author_id': ["james", "mark", "steven", "john"],
         'text': ["I am James", "I am Mark", " I am Steven", "I am John"],
         "created_at": [pd.Timestamp('2017-01-01T01'),
                        pd.Timestamp('2017-01-01T02'),
                        pd.Timestamp('2017-01-01T03'),
                        pd.Timestamp('2017-01-01T04')]}
    df = pd.DataFrame(data=d)
    manager = TreeManager(df)
    manager.initialize_trees()  # creates one tree
    test_tree = manager.random()
    return test_tree


def get_test_manager() -> TreeManager:
    d = {'tree_id': [1] * 4,
         'post_id': [1, 2, 3, 4],
         'parent_id': [None, 1, 2, 1],
         'author_id': ["james", "mark", "steven", "john"],
         'text': ["I am James", "I am Mark", " I am Steven", "I am John"],
         "created_at": [pd.Timestamp('2017-01-01T01'),
                        pd.Timestamp('2017-01-01T02'),
                        pd.Timestamp('2017-01-01T03'),
                        pd.Timestamp('2017-01-01T04')]}
    d2 = d.copy()
    d2["tree_id"] = [2] * 4
    d2['parent_id'] = [None, 1, 2, 3]
    d3 = d.copy()
    d3["tree_id"] = [3] * 4
    d3['parent_id'] = [None, 1, 1, 1]
    # a case where an author answers himself
    d4 = d.copy()
    d4["tree_id"] = [4] * 4
    d4["author_id"] = ["james", "james", "james", "john"]

    d5 = d.copy()
    d5["tree_id"] = [5] * 4
    d5['parent_id'] = [None, 1, 2, 3]
    d5["author_id"] = ["james", "james", "james", "john"]

    df1 = pd.DataFrame(data=d)
    df2 = pd.DataFrame(data=d2)
    df3 = pd.DataFrame(data=d3)
    df4 = pd.DataFrame(data=d4)
    df5 = pd.DataFrame(data=d5)
    df_list = [df1, df2, df3, df4, df5]
    df = pd.concat(df_list, ignore_index=True)
    manager = TreeManager(df)
    manager.initialize_trees()
    return manager


def get_social_media_trees(platform="twitter", context="test"):
    assert platform == "twitter" or platform == "reddit", "platform needs to be reddit or twitter!"
    if context == "test":
        file = "../delab_trees/data/dataset_reddit_no_text.pkl"
    else:
        file = "delab_trees/data/dataset_reddit_no_text.pkl"
    file = file.replace("reddit", platform)
    df = pd.read_pickle(file)
    manager = TreeManager(df)
    return manager.initialize_trees()
