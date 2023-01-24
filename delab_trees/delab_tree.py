from copy import copy, deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx import MultiDiGraph
from networkx.drawing.nx_pydot import graphviz_layout
from pandas import DataFrame

from delab_trees.delab_post import DelabPost, DelabPosts
from delab_trees.exceptions import GraphNotInitializedException
from delab_trees.util import get_root

TREE_IDENTIFIER = "tree_id"


class GRAPH:
    class ATTRIBUTES:
        CREATED_AT = "created_at"

    class LABELS:
        MENTIONS = "mentions"
        PARENT_OF = "parent_of"
        AUTHOR_OF = "author_of"
        ANSWERED_BY = "answered_by"

    class SUBSETS:
        TWEETS = "tweets"
        AUTHORS = "authors"


class TABLE:
    class COLUMNS:
        PARENT_ID = "parent_id"
        CREATED_AT = "created_at"
        AUTHOR_ID = "author_id"
        TEXT = "text"
        POST_ID = "post_id"
        TREE_ID = "tree_id"


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
            df = self.df[self.df['parent_id'].notna()]
            df = df.assign(label=GRAPH.LABELS.PARENT_OF)
            # print(df)
            networkx_graph = nx.from_pandas_edgelist(df, source="parent_id", target="post_id", edge_attr='label',
                                                     create_using=nx.MultiDiGraph())
            nx.set_node_attributes(networkx_graph, GRAPH.SUBSETS.TWEETS, name="subset")  # rename to posts
            # print(networkx_graph.edges(data=True))
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
        nx.set_node_attributes(networkx_graph, GRAPH.SUBSETS.AUTHORS, name="subset")  # rename to posts
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
        author_ids = set(self.df[TABLE.COLUMNS.AUTHOR_ID].tolist())

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

    def as_tree(self):
        if self.reply_graph is None:
            raise GraphNotInitializedException()
        root = get_root(self.reply_graph)
        tree = nx.bfs_tree(self.reply_graph, root)
        return tree

    def as_recursive_tree(self):
        # TODO IMPLEMENT recursive_tree conversion
        pass

    def as_merged_self_answers_graph(self):
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
        return G

    def get_conversation_flows(self):
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
