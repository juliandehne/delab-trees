import unittest

from delab_trees.delab_tree import DelabTree
from delab_trees.main import get_social_media_trees


class DuoFlowAlgorithmTestCase(unittest.TestCase):

    def setUp(self):
        self.manager = get_social_media_trees(n=10)

    def test_load_trees(self):
        n_trees_loaded = len(self.manager.trees)
        assert n_trees_loaded == 10

    def test_tree_structures(self):
        for tree_id, tree in self.manager.trees.items():
            tree2: DelabTree = tree
            tree2.as_tree()

    def test_flow_computation(self):
        test_tree = self.manager.random()
        print("loading:flow")
        flows, longest = test_tree.get_conversation_flows()
        assert len(flows[longest]) > 0

    def test_duo_flow_computation(self):
        test_tree = self.manager.random()
        flow_duo = test_tree.as_flow_duo()
        if flow_duo is not None:
            assert len(flow_duo.posts1) > 0 and len(flow_duo.posts2) > 0
            assert flow_duo.toxic_delta > 0
