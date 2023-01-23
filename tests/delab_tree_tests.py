import unittest

import pandas as pd

from delab_trees.delab_tree import DelabTree
from delab_trees.main import TreeManager


class DelabTreeConstructionTestCase(unittest.TestCase):

    def setUp(self):
        d = {'tree_id': [1, 1, 1, 1], 'post_id': [1, 2, 3, 4], 'parent_id': [None, 1, 2, 1],
             'author_id': ["james", "mark", "steven", "john"],
             'text': ["I am James", "I am Mark", " I am Steven", "I am John"]}
        d2 = d.copy()
        d2["tree_id"] = [2, 2, 2, 2]
        d2['parent_id'] = [None, 1, 2, 3]
        d3 = d.copy()
        d3["tree_id"] = [3, 3, 3, 3]
        d3['parent_id'] = [None, 1, 1, 1]

        df1 = pd.DataFrame(data=d)
        df2 = pd.DataFrame(data=d2)
        df3 = pd.DataFrame(data=d3)
        df_list = [df1, df2, df3]
        self.df = pd.concat(df_list, ignore_index=True)
        self.manager = TreeManager(self.df)

        # self.test_df =

    def test_load_trees(self):
        # tests if the dataframes is loaded correctly as multiple trees
        self.manager.initialize_trees()
        assert len(self.manager.trees) == 3
        n_graph = self.manager.trees[1].graph
        assert n_graph is not None
        assert len(n_graph.edges()) == 3

    def test_tree_metrics(self):
        test_tree: DelabTree = self.manager.initialize_trees().random()
        assert test_tree.total_number_of_posts() == 4
        assert test_tree.avg_branching_factor() > 0


if __name__ == '__main__':
    unittest.main()
