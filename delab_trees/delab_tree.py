import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx import MultiDiGraph
from networkx.drawing.nx_pydot import graphviz_layout
from pandas import DataFrame

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
        CREATED_AT = "created_at"
        AUTHOR_ID = "author_id"
        TEXT = "text"
        POST_ID = "post_id"
        TREE_ID = "tree_id"


class DelabTree:

    def __init__(self, df: pd.DataFrame):
        self.df: DataFrame = df
        self.reply_graph: MultiDiGraph = None
        self.author_graph: MultiDiGraph = None

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
        posts_df = self.df[["post_id", "author_id", "created_at", "parent_id"]]
        posts_df.sort_values(by="created_at", inplace=True)
        posts = None # TODO
        to_delete_list = []
        to_change_map = {}
        for tweet in posts:
            # if tweet.twitter_id == 82814624:
            #    print("testing 1")

            # we are not touching the root
            if tweet.tn_parent is None:
                continue
            # if a tweet is merged, ignore
            if tweet.twitter_id in to_delete_list:
                continue
            # if a tweet shares the author with its parent, deleted it
            if tweet.author_id == tweet.tn_parent.author_id:
                to_delete_list.append(tweet.twitter_id)
            # if the parent has been deleted, find the next available parent
            else:
                current = tweet
                while current.tn_parent.twitter_id in to_delete_list:
                    # we can make this assertion because we did not delete the root
                    assert current.tn_parent is not None
                    current = current.tn_parent
                if current.twitter_id != tweet.twitter_id:
                    to_change_map[tweet.twitter_id] = current.tn_parent.twitter_id

    def paint_reply_graph(self):
        tree = self.as_tree()
        pos = graphviz_layout(tree, prog="twopi")
        # add_attributes_to_plot(conversation_graph, pos, tree)
        nx.draw_networkx_labels(tree, pos)
        nx.draw(tree, pos)
        plt.show()
