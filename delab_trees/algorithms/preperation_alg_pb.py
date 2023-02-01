import logging

import networkx as nx
import pandas as pd

from delab_trees.delab_tree import DelabTree
from delab_trees.util import get_root
from .preperation_alg_rb import prepare_row_analysis, compute_reply_features, compute_timedelta_feature, \
    compute_root_distance_feature, compute_follower_features

from delab_trees.delab_post import DelabPost

logger = logging.getLogger(__name__)

"""
the same idea as the author presence but now the current tweet is also the beam node
- it is a multi class classification problem with the different authors as the classes,
  y is 1 if it is the author to be classified to have vision of
- sample is the tree structure of the whole reply tree 
 (a author answering one or two later may still be informative) of having seen the tweet
"""


def prepare_pb_data(manager):
    records = []
    for tree_id, tree in manager.trees.items():
        tree: DelabTree = tree
        reply_graph = tree.as_reply_graph()

        root_node = get_root(reply_graph)
        conversation_graph = tree.as_author_graph()
        follower_Graph = nx.MultiDiGraph()
        posts = tree.as_post_list()

        for post in posts:
            row_dict = calculate_forward_row(post, reply_graph, follower_Graph, conversation_graph, root_node)
            # empty dictionaries evaluate to false
            records += row_dict

    df = pd.DataFrame.from_records(records)
    df.fillna(0, inplace=True)
    return df


def calculate_forward_row(post: DelabPost, reply_graph: nx.DiGraph, follower_graph: nx.MultiDiGraph,
                          conversation_graph: nx.MultiDiGraph, root_node: int):
    """
    :param root_node:
    :param reply_graph:
    :param conversation_graph: directed graph that represents the conversation tree, node ids are twitter ids
    :param follower_graph: directed graph that represents the follower structures, node ids are the author ids
    :param post:
    :return: a dictionary of the tweet history containing the column names as keys and the features as values
    """
    conversation_depth, path_dict, reply_nodes, result_of_results, root_path_dict, row_node_author_id, row_node_id = \
        prepare_row_analysis(reply_graph, root_node, post)

    for current_node_id, current_node_attr in reply_nodes:
        result = {}
        current_node_timestamp = current_node_attr["created_at"]

        if row_node_id != current_node_id:
            compute_reply_features(path_dict, current_node_id, result, row_node_id, conversation_depth)
            compute_timedelta_feature(current_node_timestamp, result, post)
            compute_root_distance_feature(root_path_dict, current_node_id, result, root_node, row_node_id,
                                          conversation_depth)
            result["current"] = post.post_id
            result["beam_node"] = current_node_id
            compute_follower_features(conversation_graph, current_node_id, follower_graph, result,
                                      row_node_author_id)
            result["conversation_id"] = post.tree_id
            result["author"] = post.author_id
            result["current_time"] = post.created_at
            result["beam_node_time"] = current_node_timestamp
            compute_previous_posts_feature(conversation_graph, current_node_id, result, post)

        if result:
            result_of_results.append(result)

    return result_of_results


def compute_previous_posts_feature(conversation_graph, current_node_id, result, post):
    in_edges = conversation_graph.in_edges(current_node_id)
    for source, target in in_edges:
        if source == post.author_id:
            all_path = nx.all_simple_paths(conversation_graph, current_node_id, post.post_id)
            for path in all_path:
                result["same_author_path_{}".format(len(path) - 1)] = 1
