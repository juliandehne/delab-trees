import networkx as nx

from delab_trees.exceptions import NotATreeException


def get_root(conversation_graph: nx.DiGraph):  # tree rooted at 0
    """
    :param conversation_graph:
    :return: the root node of a nx graph that is a tree
    """
    roots = [n for n, d in conversation_graph.in_degree() if d == 0]
    if len(roots) != 1:
        raise NotATreeException()
    return roots[0]

