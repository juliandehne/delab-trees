# functions for testing below
import os

from delab_trees import TreeManager
from delab_trees.delab_tree import DelabTree
import pandas as pd


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

    # not connected
    d6 = d.copy()
    d6["tree_id"] = [6] * 4
    d6['parent_id'] = [None, 1, 42, 3]
    d6["author_id"] = ["james", "hannah", "jana", "john"]

    # contains cycle
    d7 = d.copy()
    d7["tree_id"] = [7] * 4
    d7['post_id'] = [1, 2, 3, 2]
    d7['parent_id'] = [None, 1, 2, 2]
    d7["author_id"] = ["james", "hannah", "jana", "john"]

    df1 = pd.DataFrame(data=d)
    df2 = pd.DataFrame(data=d2)
    df3 = pd.DataFrame(data=d3)
    df4 = pd.DataFrame(data=d4)
    df5 = pd.DataFrame(data=d5)
    df6 = pd.DataFrame(data=d6)
    df7 = pd.DataFrame(data=d7)

    # df_list = [df1, df2, df3, df4, df5, df6, df7]
    # TODO implement tree with cycles
    df_list = [df1, df2, df3, df4, df5, df6]
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
    manager = TreeManager(df, n=n)
    return manager
