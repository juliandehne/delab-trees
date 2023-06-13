import unittest

from delab_trees.delab_tree import DelabTree
from delab_trees.test_data_manager import get_social_media_trees


class RBAlgorithmsTestCase(unittest.TestCase):

    def setUp(self):
        self.manager = get_social_media_trees(n=10)

    def test_load_trees(self):
        assert len(self.manager.trees) == 10

    def test_rb_algorithm(self):
        # tree: DelabTree = self.manager.trees[1]
        # rb_vision = self.manager.get_rb_vision(tree)
        # assert rb_vision["steven"] is not None
        pass

