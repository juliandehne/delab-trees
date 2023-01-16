import unittest

import pandas as pd
from pandas import DataFrame

from delab_trees.delab_trees import DelabTrees


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
        self.delab = DelabTrees(self.df)

        # self.test_df =

    def test_load_trees(self):
        """Test 0 multiplied by 2"""
        self.delab.initialize_trees()
        assert len(self.delab.trees) == 3
        n_graph = self.delab.trees[1].graph
        assert n_graph is not None
        assert len(n_graph.edges()) == 3


if __name__ == '__main__':
    unittest.main()
