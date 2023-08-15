import unittest

from delab_trees.delab_tree import DelabTree
from delab_trees.test_data_manager import get_test_manager, get_test_tree


class DelabTreeConstructionTestCase(unittest.TestCase):

    def setUp(self):
        self.manager = get_test_manager()

    def test_load_single_tree(self):
        test_tree: DelabTree = get_test_tree()
        assert test_tree.total_number_of_posts() == 4
        assert test_tree.average_branching_factor() > 0

    def test_load_trees(self):
        # tests if the dataframes is loaded correctly as multiple trees
        assert len(self.manager.trees) == 6
        n_graph = self.manager.trees[1].reply_graph
        assert n_graph is not None
        assert len(n_graph.edges()) == 3

    def test_tree_metrics(self):
        test_tree: DelabTree = self.manager.random()
        assert test_tree.total_number_of_posts() == 4
        assert test_tree.average_branching_factor() > 0
        test_tree2: DelabTree = self.manager.trees[1]
        assert test_tree2.root_dominance() == 0.25
        test_tree3: DelabTree = self.manager.trees[4]
        assert test_tree3.root_dominance() == 0.75
        # print("\n\nNOTES: ")
        # print("the branching weight is {}".format(test_tree.branching_weight()))
        # print("the avg branching factor is {}".format(test_tree.average_branching_factor()))

    def test_author_graph(self):
        tree: DelabTree = self.manager.trees[1]
        author_graph = tree.as_author_graph()
        assert len(author_graph.edges()) == 7

    def test_merge_subsequent_graph(self):
        tree: DelabTree = self.manager.trees[4]
        merged_graph = tree.as_merged_self_answers_graph(as_delab_tree=False)
        assert len(merged_graph.edges()) == 1
        # print(merged_graph.edges(data=True))
        tree2: DelabTree = self.manager.trees[5]
        merged_graph2: DelabTree = tree2.as_merged_self_answers_graph(as_delab_tree=False)
        assert len(merged_graph2.edges()) == 1

    def test_flow_computation(self):
        tree: DelabTree = self.manager.trees[4]
        flow_dict, name_of_longest = tree.get_conversation_flows()
        assert len(flow_dict[name_of_longest]) == 3

    def test_author_centrality(self):
        tree: DelabTree = self.manager.trees[1]
        measures = tree.get_author_metrics()
        assert measures is not None

    def test_author_baseline_vision(self):
        tree: DelabTree = self.manager.trees[1]
        measures = tree.get_author_metrics()
        author_measures_steven = measures["steven"]
        author_measures_mark = measures["mark"]
        # assert author_measures_steven.baseline_author_vision > author_measures_mark.baseline_author_vision
        # TODO: Check plausibility of baseline vision calculation
        assert author_measures_steven.baseline_author_vision > 0

    def test_validated(self):
        assert not self.manager.validate(verbose=False)

    def test_attached_orphans(self):
        tree: DelabTree = self.manager.trees[6]
        tree = tree.as_attached_orphans(as_delab_tree=True)
        tree.validate(verbose=False)
        assert len(tree.as_reply_graph().nodes()) == 4

    def test_remove_cycles(self):
        """
        assert not self.manager.validate(verbose=False)
        tree_manager = self.manager.remove_cycles()
        tree: DelabTree = tree_manager.trees[7]
        tree.validate(verbose=False)
        assert len(tree.as_reply_graph().nodes()) == 3
        """
        pass


if __name__ == '__main__':
    unittest.main()
