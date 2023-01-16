import networkx as nx
import pandas as pd

from delab_trees.exceptions import GraphNotInitializedException


class DelabTree:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.graph = None

    def avg_branching_factor(self):
        if self.graph is None:
            raise GraphNotInitializedException()
        # TODO write code for avg_branching_factor
        nx_tree = self.df.index
        return None

    def total_number_of_posts(self):
        return len(self.df.index)

    def initialize_networkx_tree(self):
        df = self.df[self.df['parent_id'].notna()]
        networkx_graph = nx.from_pandas_edgelist(df, source="parent_id", target="post_id", edge_attr=True,
                                                 create_using=nx.MultiDiGraph())
        self.graph = networkx_graph
        return networkx_graph
