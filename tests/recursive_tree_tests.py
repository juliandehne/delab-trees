import unittest

from delab_trees.delab_tree import DelabTree
from delab_trees.test_data_manager import get_test_manager
from delab_trees.test_data_manager import get_example_conversation_tree


class DelabTreeConstructionTestCase(unittest.TestCase):

    def setUp(self):
        # self.manager = get_test_manager()
        pass

    def test_load_recursive_Tree(self):
        # hello
        example_tree = get_example_conversation_tree()
        tree = DelabTree.from_recursive_tree(example_tree)
        assert tree.validate()
