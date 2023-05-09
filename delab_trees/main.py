import inspect
import logging
import os
import pickle
from random import choice
from statistics import mean

import numpy as np
import pandas as pd
from tqdm import tqdm

from delab_trees.constants import TREE_IDENTIFIER
from delab_trees.delab_author_metric import AuthorMetric
from delab_trees.delab_tree import DelabTree
from delab_trees.preperation_alg_pb import prepare_pb_data
from delab_trees.preperation_alg_rb import prepare_rb_data
from delab_trees.training_alg_pb import train_pb
from delab_trees.training_alg_rb import train_rb
import multiprocessing
from multiprocessing import Pool
from functools import partial

logger = logging.getLogger(__name__)


class TreeManager:

    def __init__(self, df, n=None):
        self.num_cpus = max(multiprocessing.cpu_count() - 2, 1)
        self.trees = {}
        self.df = df
        self.__pre_process_df()
        self.__initialize_trees(n)

    def internal_hello_world(self):
        print("hello world internal 5" + str(self.df.shape))

    def __pre_process_df(self):
        """
        convert float and int ids to str
        :return:
        """

        if self.df["parent_id"].dtype != "object" and self.df["post_id"].dtype != "object":
            df_parent_view = self.df.loc[:, "parent_id"]
            df_post_view = self.df.loc[:, "post_id"]
            self.df.loc[:, "parent_id"] = df_parent_view.astype(float).astype(str)
            self.df.loc[:, "post_id"] = df_post_view.astype(float).astype(str)
        else:
            assert self.df["parent_id"].dtype == "object" and self.df[
                "post_id"].dtype == "object", "post_id and parent_id need to be both float or str"

    def __initialize_trees(self, n=None):
        print("loading data into manager and converting table into trees...")
        grouped_by_tree_ids = {k: v for k, v in self.df.groupby(TREE_IDENTIFIER)}

        if n is not None and n < self.num_cpus:
            # cannot parallelize if the n of wanted trees is less then the cpus to be used
            trees_result = create_trees_from_grouped(n, grouped_by_tree_ids)
            self.trees = trees_result
        else:

            # compute the trees in parallel, n will be ignored until later
            # num_splits = self.num_cpus
            num_splits = len(grouped_by_tree_ids)

            # Split the dictionary into a list of sub-dictionaries
            sub_dicts = np.array_split(list(grouped_by_tree_ids.items()), num_splits)

            # Convert each sub-list into a dictionary and put them in a list
            list_of_groupings = [dict(sub_dict) for sub_dict in sub_dicts]

            with Pool(self.num_cpus) as p:
                # use the imap_unordered function to parallelize the loop
                for tree_group in tqdm(p.imap_unordered(create_trees_from_grouped_helper, list_of_groupings),
                                       total=num_splits):
                    self.trees.update(tree_group)

        return self

    def single(self) -> DelabTree:
        assert len(self.trees.keys()) == 1, "There needs to be exactly one tree in the manager!"
        first_key = list(self.trees.keys())[0]
        return self.trees[first_key]

    def random(self) -> DelabTree:
        assert len(self.trees.keys()) >= 1
        return self.trees[choice(list(self.trees.keys()))]

    def validate(self, verbose=True):
        assert self.df["post_id"].notnull().all(), "post_ids should not be null"

        parents = set(list(self.df["parent_id"]))
        posts = set(list(self.df["post_id"]))
        missing_parents = parents - posts.intersection(parents)

        if len(missing_parents) != 1:
            print("the following parents are not contained in the id column:", list(missing_parents)[:5])

        assert len(missing_parents) == 1, "all parent ids except NAN should be contained in the post id column"

        tree: DelabTree
        for tree_id, tree in tqdm(self.trees.items()):
            is_valid = tree.validate(verbose)
            assert is_valid, "Tree with id {} is not valid".format(tree_id)
            if not is_valid:
                break
        return True

    def get_mean_author_metrics(self):
        sig = inspect.signature(AuthorMetric.__init__)
        parameters = {key: value for key, value in sig.parameters.items() if key not in ['self', 'args', 'kwargs']}
        metric_names = list(parameters.keys())

        assert len(metric_names) > 0, "AuthorMetric should contain one user written attr at least"

        treeid2metrics = {}
        for tree_id, tree in self.trees.items():
            # m_author = tree_id2m_author_id[tree_id]
            tree_metrics = tree.get_author_metrics()
            treeid2metrics[tree_id] = tree_metrics

        metrics = treeid2metrics.values()
        assert len(metrics) > 0

        def get_mean_metric(metric_name):
            katz_centralise = []
            for author2metric in metrics:
                a_metrics = author2metric.values()
                for a_metric in a_metrics:
                    single_measure = getattr(a_metric, metric_name)
                    katz_centralise.append(single_measure)
            m_katz_centrality = mean(katz_centralise)
            return m_katz_centrality

        result = {name: get_mean_metric(name) for name in metric_names}
        return result

    def remove_invalid(self):
        to_remove = []
        for tree_id, tree in tqdm(self.trees.items()):
            if not tree.validate(verbose=False):
                to_remove.append(tree_id)
        for elem in to_remove:
            self.trees.pop(elem)
            # self.df = self.df.drop(self.df[self.df['tree_id'] == elem].index)
        self.df.drop(self.df['tree_id'].isin(to_remove).index)

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
    # creates one tree
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
    return manager


def get_social_media_trees(platform="twitter", n=None, context="production") -> TreeManager:
    assert platform == "twitter" or platform == "reddit", "platform needs to be reddit or twitter!"
    if context == "test":
        file = "../delab_trees/data/dataset_twitter_no_text.pkl"
        # file = "/home/dehne/PycharmProjects/delab/scriptspy/dataset_twitter_no_text.pkl"
    else:
        this_dir, this_filename = os.path.split(__file__)
        file = os.path.join(this_dir, 'data/dataset_twitter_no_text.pkl')
    file = file.replace("reddit", platform)
    df = pd.read_pickle(file)
    manager = TreeManager(df, n)
    return manager


def hello_world():
    print("hello world 9")


def create_trees_from_grouped(n_trees, groupings):
    trees = {}
    counter = 0
    for tree_id, df in groupings.items():
        counter += 1
        if n_trees is not None and counter > n_trees:
            break
        tree = DelabTree(df)
        trees[tree_id] = tree
    return trees


def create_trees_from_grouped_helper(groupings):
    return create_trees_from_grouped(None, groupings)
