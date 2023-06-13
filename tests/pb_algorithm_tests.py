import unittest

from delab_trees.delab_tree import DelabTree
from delab_trees.test_data_manager import get_social_media_trees


class PBAlgorithmsTestCase(unittest.TestCase):

    def setUp(self):
        self.manager = get_social_media_trees(n=10)

    def test_load_trees(self):
        assert len(self.manager.trees) == 10

    def test_pb_algorithm(self):
        # tree: DelabTree = self.manager.trees[1]
        # pb_vision = self.manager.get_pb_vision(tree)
        # assert pb_vision["steven"] is not None
        pass

    def test_duo_flow_computation(self):
        pass
