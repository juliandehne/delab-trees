import itertools

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


def get_all_reply_paths(conversation_graph: nx.DiGraph, min_path_length, required_max_path_length):
    """
    Get all reply paths that fall in the length window
    :param conversation_graph:
    :param min_path_length:
    :param required_max_path_length:
    :return:
    """
    G = conversation_graph
    all_paths = []
    nodes_combs = itertools.combinations(G.nodes, 2)
    for source, target in nodes_combs:
        paths = nx.all_simple_paths(G, source=source, target=target, cutoff=required_max_path_length)

        for path in paths:
            if path not in all_paths and path[::-1] not in all_paths and len(path) >= min_path_length:
                all_paths.append(path)
    return all_paths


def get_path(post_id, conversation_graph: nx.DiGraph, min_path_length=3, required_max_path_length=4):
    paths = get_all_reply_paths(conversation_graph, min_path_length, required_max_path_length)
    current_best_path_index = None
    current_best_score = 0
    index_count = 0
    for path in paths:
        if post_id in path:
            p_index = path.index(post_id)
            previous_tweets = p_index
            following_tweets = len(path) - p_index - 1
            middleness_score = min(previous_tweets, following_tweets) - abs(previous_tweets - following_tweets)
            if middleness_score > current_best_score:
                current_best_path_index = index_count
            current_best_score = max(current_best_score, middleness_score)
        index_count += 1
    if current_best_path_index is None:
        return None
    return paths[current_best_path_index]
