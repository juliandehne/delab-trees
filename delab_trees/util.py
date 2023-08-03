import itertools

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from delab_trees.exceptions import NotATreeException
from networkx.drawing.nx_pydot import graphviz_layout


def get_root(conversation_graph: nx.DiGraph):  # tree rooted at 0
    """
    :param conversation_graph:
    :return: the root node of a nx graph that is a tree
    """
    roots = get_all_roots(conversation_graph)
    if len(roots) != 1:
        raise NotATreeException(message=roots)
    return roots[0]


def get_all_roots(conversation_graph: nx.DiGraph):  # tree rooted at 0
    """
    :param conversation_graph:
    :return: the root nodes of a nx graph (no incoming edges)
    """
    roots = [n for n, d in conversation_graph.in_degree() if d == 0]
    return roots


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


def convert_float_ids_to_readable_str(string_num):
    if isinstance(string_num, str):
        return string_num

    # convert the string to a floating-point number
    float_num = float(string_num)

    # convert the floating-point number to an integer
    int_num = int(float_num)

    # convert the integer back to a string
    str_num = str(int_num)

    return str_num


def paint_bipartite_author_graph(G2, root_node):
    # Specify the edges you want here
    red_edges = [(source, target, attr) for source, target, attr in G2.edges(data=True) if
                 attr['label'] == 'author_of']
    # edge_colours = ['black' if edge not in red_edges else 'red'
    #                for edge in G2.edges()]
    black_edges = [edge for edge in G2.edges(data=True) if edge not in red_edges]
    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    paint_bipartite(G2, black_edges, red_edges, root_node=root_node)


def paint_bipartite(G2, black_edges, red_edges, root_node):
    # pos = nx.multipartite_layout(G2)

    pos = graphviz_layout(G2, prog="twopi", root=root_node)
    nx.draw_networkx_nodes(G2, pos, node_size=400)
    nx.draw_networkx_labels(G2, pos)
    nx.draw_networkx_edges(G2, pos, edgelist=red_edges, edge_color='red', arrows=True)
    nx.draw_networkx_edges(G2, pos, edgelist=black_edges, arrows=True)
    plt.show()


def pd_is_nan(parent_id):
    if isinstance(parent_id, pd.Series):
        return parent_id.apply(lambda x: pd_is_nan(x))
    result = parent_id is None or parent_id == 'nan' or pd.isna(parent_id) or parent_id == "NA" or parent_id == "None"
    if result:
        pass
    return result


def get_missing_parents(df):
    parents = set(list(df["parent_id"]))
    posts = set(list(df["post_id"]))
    intersection = posts.intersection(parents)
    missing_parent_ids = parents - intersection
    missing_parent_ids_no_nan = list(filter(lambda x: not (pd_is_nan(x)), missing_parent_ids))
    return missing_parent_ids_no_nan
