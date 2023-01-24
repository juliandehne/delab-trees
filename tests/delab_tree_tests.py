import unittest

import pandas as pd

from delab_trees.delab_tree import DelabTree
from delab_trees.main import TreeManager


class DelabTreeConstructionTestCase(unittest.TestCase):

    def setUp(self):
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
        self.df = pd.concat(df_list, ignore_index=True)
        self.manager = TreeManager(self.df)
        self.manager.initialize_trees()

    def test_load_trees(self):
        # tests if the dataframes is loaded correctly as multiple trees
        assert len(self.manager.trees) == 4
        n_graph = self.manager.trees[1].reply_graph
        assert n_graph is not None
        assert len(n_graph.edges()) == 3

    def test_tree_metrics(self):
        test_tree: DelabTree = self.manager.random()
        assert test_tree.total_number_of_posts() == 4
        assert test_tree.average_branching_factor() > 0
        # print("\n\nNOTES: ")
        # print("the branching weight is {}".format(test_tree.branching_weight()))
        # print("the avg branching factor is {}".format(test_tree.average_branching_factor()))

    def test_author_graph(self):
        tree: DelabTree = self.manager.trees[1]
        author_graph = tree.as_author_graph()
        assert len(author_graph.edges()) == 7

    def test_merge_subsequent_graph(self):
        tree: DelabTree = self.manager.trees[4]
        merged_graph = tree.as_merged_self_answers_graph()
        assert len(merged_graph.edges()) == 1
        # print(merged_graph.edges(data=True))
        tree2: DelabTree = self.manager.trees[5]
        merged_graph2 = tree2.as_merged_self_answers_graph()
        assert len(merged_graph2.edges()) == 1


if __name__ == '__main__':
    unittest.main()
