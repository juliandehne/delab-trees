import unittest

from delab_trees.delab_tree import DelabTree
from delab_trees.test_data_manager import get_test_manager
from delab_trees.test_data_manager import get_example_conversation_tree


class DelabTreeConstructionTestCase(unittest.TestCase):

    def setUp(self):
        self.manager = get_test_manager()
        print("setup")

    def test_load_recursive_tree(self):
        # hello
        self.example_tree = get_example_conversation_tree()
        tree = DelabTree.from_recursive_tree(self.example_tree)
        assert tree.validate()
        print(tree.as_recursive_tree().to_string())

    def test_convert_to_recursive(self):
        tree: DelabTree = self.manager.trees[1]
        root_node = tree.as_recursive_tree()
        assert root_node.compute_max_path_length() == 2
