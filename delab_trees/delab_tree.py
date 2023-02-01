from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx import MultiDiGraph
from networkx.drawing.nx_pydot import graphviz_layout
from pandas import DataFrame

from delab_trees.constants import TABLE, GRAPH
from delab_trees.delab_author_metric import AuthorMetric
from delab_trees.delab_post import DelabPosts, DelabPost
from delab_trees.exceptions import GraphNotInitializedException
from delab_trees.util import get_root


class DelabTree:

    def __init__(self, df: pd.DataFrame):
        self.df: DataFrame = deepcopy(df)
        self.reply_graph: MultiDiGraph = None
        self.author_graph: MultiDiGraph = None
        self.conversation_id = self.df.iloc[0][TABLE.COLUMNS.TREE_ID]
        # print("initialized")

    def branching_weight(self):
        if self.reply_graph is None:
            raise GraphNotInitializedException()
        return nx.tree.branching_weight(self.as_tree())

    def average_branching_factor(self):
        result = 2 * self.reply_graph.number_of_edges() / self.reply_graph.number_of_nodes()
        return result

    def total_number_of_posts(self):
        return len(self.df.index)

    def as_reply_graph(self):
        if self.reply_graph is None:
            node2creation = self.df.set_index(TABLE.COLUMNS.POST_ID).to_dict()[TABLE.COLUMNS.CREATED_AT]
            df = self.df[self.df[TABLE.COLUMNS.PARENT_ID].notna()]
            df = df.assign(label=GRAPH.LABELS.PARENT_OF)
            networkx_graph = nx.from_pandas_edgelist(df,
                                                     source=TABLE.COLUMNS.PARENT_ID,
                                                     target=TABLE.COLUMNS.POST_ID,
                                                     edge_attr='label',
                                                     create_using=nx.MultiDiGraph())
            nx.set_node_attributes(networkx_graph, GRAPH.SUBSETS.TWEETS, name="subset")  # rename to posts
            nx.set_node_attributes(networkx_graph, node2creation, name=TABLE.COLUMNS.CREATED_AT)
            self.reply_graph = networkx_graph
        return self.reply_graph

    def as_author_graph(self):
        """
        This computes the combined reply graph with the author_of relations included.
        :return:
        """
        if self.reply_graph is None:
            self.as_reply_graph()
        if self.author_graph is not None:
            return self.author_graph
        df = self.df.assign(label=GRAPH.LABELS.AUTHOR_OF)
        # print(df)
        networkx_graph = nx.from_pandas_edgelist(df, source="author_id", target="post_id", edge_attr='label',
                                                 create_using=nx.MultiDiGraph())
        author2authorlabel = dict([(author_id, GRAPH.SUBSETS.AUTHORS) for author_id in self.__get_author_ids()])
        nx.set_node_attributes(networkx_graph, author2authorlabel, name="subset")  # rename to posts
        # print(networkx_graph.edges(data=True))
        self.author_graph = nx.compose(self.reply_graph, networkx_graph)
        return self.author_graph

    def as_author_interaction_graph(self):
        """
        This computes the projected graph from the reply graph to the who answered whom graph (different nodes).
        This could be considered a unweighted bipartite projection.
        :return:
        """
        G = self.as_author_graph()
        # assuming the dataframe and the reply graph are two views on the same data!
        author_ids = self.__get_author_ids()

        G2 = nx.DiGraph()
        G2.add_nodes_from(author_ids)

        for a in author_ids:
            tw1_out_edges = G.out_edges(a, data=True)
            for _, tw1, out_attr in tw1_out_edges:
                tw2_out_edges = G.out_edges(tw1, data=True)
                for _, tw2, _ in tw2_out_edges:
                    in_edges = G.in_edges(tw2, data=True)
                    # since target already has a source, there can only be in-edges of type author_of
                    for reply_author, _, in_attr in in_edges:
                        if in_attr["label"] == GRAPH.LABELS.AUTHOR_OF:
                            assert reply_author in author_ids
                            if a != reply_author:
                                G2.add_edge(a, reply_author, label=GRAPH.LABELS.ANSWERED_BY)

        return G2

    def __get_author_ids(self):
        author_ids = set(self.df[TABLE.COLUMNS.AUTHOR_ID].tolist())
        return author_ids

    def as_tree(self):
        if self.reply_graph is None:
            raise GraphNotInitializedException()
        root = get_root(self.reply_graph)
        tree = nx.bfs_tree(self.reply_graph, root)
        return tree

    def as_post_list(self) -> list[DelabPost]:
        return DelabPosts.from_pandas(self.df)

    def as_recursive_tree(self):
        # TODO IMPLEMENT recursive_tree conversion
        # The recursive Tree has the tostring and toxml implemented
        pass

    def as_merged_self_answers_graph(self, return_deleted=False):
        posts_df = self.df[[TABLE.COLUMNS.POST_ID,
                            TABLE.COLUMNS.AUTHOR_ID,
                            TABLE.COLUMNS.CREATED_AT,
                            TABLE.COLUMNS.PARENT_ID]]
        posts_df = posts_df.sort_values(by=TABLE.COLUMNS.CREATED_AT)
        to_delete_list = []
        to_change_map = {}
        for row_index in posts_df.index.values:
            # we are not touching the root
            author_id, parent_author_id, parent_id, post_id = self.__get_table_row_as_names(posts_df, row_index)
            if parent_id is None or np.isnan(parent_id):
                continue
            # if a tweet is merged, ignore
            if post_id in to_delete_list:
                continue
            # if a tweet shares the author with its parent, deleted it
            if author_id == parent_author_id:
                to_delete_list.append(post_id)
            # if the parent has been deleted, find the next available parent
            else:
                current = row_index
                moving_post_id = deepcopy(post_id)
                moving_parent_id = deepcopy(parent_id)
                moving_parent_author_id = deepcopy(parent_author_id)
                while moving_parent_id in to_delete_list:
                    # we can make this assertion because we did not delete the root
                    moving_author_id, moving_parent_author_id, moving_parent_id, moving_post_id = \
                        self.__get_table_row_as_names(posts_df, current)
                    assert moving_parent_id is not None
                    current = posts_df.index[posts_df[TABLE.COLUMNS.POST_ID] == moving_parent_id].values[0]
                if moving_post_id != post_id:
                    to_change_map[post_id] = moving_parent_id

        # constructing the new graph
        G = nx.DiGraph()
        edges = []
        row_indexes2 = [row_index for row_index in posts_df.index
                        if posts_df.loc[row_index][TABLE.COLUMNS.POST_ID] not in to_delete_list]
        post_ids = list(posts_df.loc[row_indexes2][TABLE.COLUMNS.POST_ID])
        for row_index2 in row_indexes2:
            author_id, parent_author_id, parent_id, post_id = self.__get_table_row_as_names(posts_df, row_index2)
            if parent_id is not None and not np.isnan(parent_id):
                # if parent_id not in post_ids and parent_id not in to_delete_list:
                #     print("conversation {} has no root_node".format(self.conversation_id))
                if post_id in to_change_map:
                    new_parent = to_change_map[post_id]
                    if new_parent in post_ids:
                        edges.append((new_parent, post_id))
                    else:
                        G.remove_node(post_id)
                else:
                    edges.append((parent_id, post_id))

        assert len(edges) > 0, "there are no edges for conversation {}".format(self.conversation_id)
        G.add_edges_from(edges, label=GRAPH.LABELS.PARENT_OF)
        nx.set_node_attributes(G, GRAPH.SUBSETS.TWEETS, name="subset")
        # return G, to_delete_list, changed_nodes
        print("removed {} and changed {}".format(to_delete_list, to_change_map))
        if return_deleted:
            return G, to_delete_list, to_change_map
        return G

    def get_conversation_flows(self):
        """
        computes all flows (paths that lead from root to leaf) in the reply tree
        :rtype: object
        :return:
        """
        reply_tree = self.reply_graph
        root = get_root(reply_tree)
        leaves = [x for x in reply_tree.nodes() if reply_tree.out_degree(x) == 0]
        flows = []
        flow_dict = {}
        for leaf in leaves:
            paths = nx.all_simple_paths(reply_tree, root, leaf)
            flows.append(next(paths))
        for flow in flows:
            flow_name = str(flow[0]) + "_" + str(flow[-1])
            flow_tweets_frame = self.df[self.df[TABLE.COLUMNS.POST_ID].isin(flow)]
            flow_tweets = DelabPosts.from_pandas(flow_tweets_frame)
            flow_dict[flow_name] = flow_tweets

        name_of_longest = max(flow_dict, key=lambda x: len(set(flow_dict[x])))
        return flow_dict, name_of_longest

    def get_author_metrics(self):
        result = {}
        author_interaction_graph = self.as_author_interaction_graph()
        katz_centrality = nx.katz_centrality(author_interaction_graph)
        baseline_author_vision = self.get_baseline_author_vision()
        try:
            betweenness_centrality = nx.betweenness_centrality(author_interaction_graph)
        except ValueError:
            betweenness_centrality = {}

        author_ids = self.__get_author_ids()
        for author_id in author_ids:
            a_closeness_centrality = nx.closeness_centrality(author_interaction_graph, author_id)
            a_betweenness_centrality = betweenness_centrality.get(author_id, None)
            a_katz_centrality = katz_centrality.get(author_id, None)
            a_baseline_author_vision = baseline_author_vision.get(author_id, None)
            metric = AuthorMetric(a_closeness_centrality,
                                  a_betweenness_centrality,
                                  a_katz_centrality,
                                  a_baseline_author_vision)
            result[author_id] = metric

        return result

    def get_baseline_author_vision(self):
        author2baseline = {}
        author_interaction_graph, to_delete_list, to_change_map = self.as_merged_self_answers_graph(return_deleted=True)
        root = get_root(author_interaction_graph)
        post2author, author2posts = self.__get_author_post_map()
        for node in to_delete_list:
            post2author.pop(node, None)
        for author in author2posts.keys():
            n_posts = len(author2posts[author])
            root_distance_measure = 0
            reply_vision_measure = 0
            for post in author2posts[author]:
                if post in to_delete_list:
                    continue
                if post == root:
                    root_distance_measure += 1
                else:
                    path = next(nx.all_simple_paths(author_interaction_graph, root, post))
                    root_distance = len(path)
                    root_distance_measure += 0.25 ** root_distance
                    reply_vision_path_measure = 0

                    reply_paths = next(nx.all_simple_paths(author_interaction_graph, root, post))
                    for previous_tweet in reply_paths:
                        if previous_tweet != post:
                            path_to_previous = nx.all_simple_paths(author_interaction_graph, previous_tweet, post)
                            path_to_previous = next(path_to_previous)
                            path_length = len(path_to_previous)
                            reply_vision_path_measure += 0.5 ** path_length
                    reply_vision_path_measure = reply_vision_path_measure / len(reply_paths)
                    reply_vision_measure += reply_vision_path_measure
            root_distance_measure = root_distance_measure / n_posts
            reply_vision_measure = reply_vision_measure / n_posts
            # author2baseline[author] = (root_distance_measure + reply_vision_measure) / 2  # un-normalized
            author2baseline[author] = reply_vision_measure  # un-normalized
            baseline = author2baseline[author]
            assert 0 <= baseline <= 1
        return author2baseline

    def __get_author_post_map(self):
        tweet2author = my_dict = self.df.set_index(TABLE.COLUMNS.POST_ID)[TABLE.COLUMNS.AUTHOR_ID].to_dict()
        inverted_dict = defaultdict(list)
        for key, value in tweet2author.items():
            inverted_dict[value].append(key)
        author2posts = dict(inverted_dict)
        return tweet2author, author2posts

    def get_single_author_metrics(self, author_id):
        return self.get_author_metrics().get(author_id, None)

    @staticmethod
    def __get_table_row_as_names(posts_df, row_index):
        post_data = posts_df.loc[row_index]
        parent_id = post_data[TABLE.COLUMNS.PARENT_ID]
        post_id = post_data[TABLE.COLUMNS.POST_ID]
        author_id = post_data[TABLE.COLUMNS.AUTHOR_ID]
        parent_author_id = None
        # if parent_id is not None and np.isnan(parent_id) is False:
        parent_author_frame = posts_df[posts_df[TABLE.COLUMNS.POST_ID] == parent_id]
        if not parent_author_frame.empty:
            parent_author_id = parent_author_frame.iloc[0][TABLE.COLUMNS.AUTHOR_ID]
        return author_id, parent_author_id, parent_id, post_id

    def paint_reply_graph(self):
        tree = self.as_tree()
        pos = graphviz_layout(tree, prog="twopi")
        # add_attributes_to_plot(conversation_graph, pos, tree)
        nx.draw_networkx_labels(tree, pos)
        nx.draw(tree, pos)
        plt.show()
