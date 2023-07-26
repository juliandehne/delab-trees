import statistics
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx import MultiDiGraph, NetworkXPointlessConcept, NetworkXNoCycle, DiGraph
from networkx.drawing.nx_pydot import graphviz_layout
from pandas import DataFrame

from delab_trees.constants import TABLE, GRAPH
from delab_trees.delab_author_metric import AuthorMetric
from delab_trees.delab_post import DelabPosts, DelabPost
from delab_trees.exceptions import GraphNotInitializedException
from delab_trees.flow_duos import compute_highest_flow_delta, FLowDuo
from delab_trees.recursive_tree.recursive_tree import TreeNode
from delab_trees.util import get_root, convert_float_ids_to_readable_str, paint_bipartite_author_graph, pd_is_nan, \
    get_missing_parents, get_all_roots

from delab_trees.recursive_tree.recursive_tree_util import solve_orphans


class DelabTree:

    def __init__(self, df: pd.DataFrame, g: MultiDiGraph = None):
        """
        Assumes the following columns as a pandas df:
        (see constants.py)
        class TABLE:
            class COLUMNS:
                PARENT_ID = "parent_id"
                CREATED_AT = "created_at"
                AUTHOR_ID = "author_id"
                TEXT = "text"
                POST_ID = "post_id"
                TREE_ID = "tree_id"
        :param df:
        """
        self.df: DataFrame = deepcopy(df)
        if g is None:
            self.reply_graph: DiGraph = self.as_reply_graph()
        else:
            self.reply_graph = g
        self.author_graph: MultiDiGraph = None
        self.conversation_id = self.df.iloc[0][TABLE.COLUMNS.TREE_ID]

    def __str__(self):
        return f"DelabTree with graph{self.reply_graph} and max_depth{self.depth()}"

    @classmethod
    def from_recursive_tree(cls, root_node: TreeNode):
        rows = root_node.to_post_list()
        df = pd.DataFrame(rows)
        return cls(df=df)

    def __update_df(self):
        post_ids = self.reply_graph.nodes
        self.df = self.df[self.df["post_id"].isin(post_ids)]

    def branching_weight(self):
        if self.reply_graph is None:
            raise GraphNotInitializedException()
        return nx.tree.branching_weight(self.as_tree())

    def average_branching_factor(self):
        result = 2 * self.reply_graph.number_of_edges() / self.reply_graph.number_of_nodes()
        return result

    def root_dominance(self):
        root_node = get_root(self.reply_graph)
        root_author = self.df[self.df[TABLE.COLUMNS.POST_ID] == root_node][TABLE.COLUMNS.AUTHOR_ID]
        root_author = list(root_author)[0]
        self.df["is_root"] = self.df[TABLE.COLUMNS.AUTHOR_ID] == root_author
        root_dominance = self.df["is_root"].sum() / len(self.df.index)
        return root_dominance

    def total_number_of_posts(self):
        return len(self.df.index)

    def depth(self):
        longest_path = nx.dag_longest_path(self.reply_graph)
        return len(longest_path)

    def as_reply_graph(self):
        df2: DataFrame = deepcopy(self.df)
        node2creation = df2.set_index(TABLE.COLUMNS.POST_ID).to_dict()[TABLE.COLUMNS.CREATED_AT]
        # df2 = df2[df2[TABLE.COLUMNS.PARENT_ID] != 'nan']
        # df2 = df2[df2[TABLE.COLUMNS.PARENT_ID].notna()]
        root_mask = df2[TABLE.COLUMNS.PARENT_ID].apply(pd_is_nan)
        df2 = df2[~root_mask]

        df2 = df2.assign(label=GRAPH.LABELS.PARENT_OF)
        networkx_graph = nx.from_pandas_edgelist(df2,
                                                 source=TABLE.COLUMNS.PARENT_ID,
                                                 target=TABLE.COLUMNS.POST_ID,
                                                 edge_attr='label',
                                                 create_using=nx.DiGraph())
        nx.set_node_attributes(networkx_graph, GRAPH.SUBSETS.TWEETS, name="subset")  # rename to posts
        nx.set_node_attributes(networkx_graph, node2creation, name=TABLE.COLUMNS.CREATED_AT)
        # draw the graph
        # nx.draw(networkx_graph)
        # plt.show()
        return networkx_graph

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
                                                 create_using=nx.DiGraph())
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
        # The recursive Tree has the tostring and toxml implemented
        df_sorted = self.df.sort_values('created_at')  # sort for better performance when inserting
        rows = df_sorted.to_dict(orient='records')
        root = None
        orphans = []
        for row in rows:
            if pd_is_nan(row[TABLE.COLUMNS.PARENT_ID]):
                root = row
            else:
                non_root_node = TreeNode(data=row, node_id=row["post_id"], parent_id=row["parent_id"],
                                         tree_id=self.conversation_id)
                orphans.append(non_root_node)
        root_node = TreeNode(data=root, node_id=root["post_id"], tree_id=self.conversation_id)
        # now solve all the orphans that have not been seen
        orphan_added = True
        while orphan_added:
            orphan_added, orphans = solve_orphans(orphans, root_node)
        return root_node

    def as_biggest_connected_tree(self, stateless=True):
        """
        if there are faulty trees you can use this to approximate the biggest tree
        :param stateless: if not true, returns the DelabTree as a new object
        :return:
        """
        # create a graph
        G = self.reply_graph

        # initialize variables to keep track of the largest tree found so far
        largest_tree_size = 0
        largest_tree = None

        # iterate over the connected components of the graph
        for component in nx.connected_components(G):
            # check if the component has the tree property
            if nx.is_tree(G.subgraph(component)):
                # calculate the size of the tree
                tree_size = nx.number_of_nodes(G.subgraph(component))
                # update the largest tree found so far
                if tree_size > largest_tree_size:
                    largest_tree_size = tree_size
                    largest_tree = component
        if not stateless:
            self.reply_graph = largest_tree
            self.__update_df()
            return self
        return largest_tree

    def as_removed_cycles(self, as_delab_tree=True, compute_arborescence=False):
        """
        remove cycles in graph and return minimum spanning arborescence
        :param compute_arborescence:
        :param as_delab_tree: if True will return a new DelabTree object instead of the edited graph
        :return:
        """
        # assert nx.is_weakly_connected(self.reply_graph), "the graph needs to be weakly connected to remove cycles"

        G = self.reply_graph
        # find all the simple cycles in the graph
        cycles = list(nx.simple_cycles(G))

        if not compute_arborescence:
            # remove the edges that belong to the cycles
            for cycle in cycles:
                for i in range(len(cycle)):
                    G.remove_edge(cycle[i], cycle[(i + 1) % len(cycle)])

        # find the minimum spanning arborescence of the graph
        else:
            G = nx.minimum_spanning_arborescence(G)

        if not as_delab_tree:
            return G
        else:
            new_df = self.df[self.df[["parent_id", "post_id"]]
                .apply(tuple, axis=1).isin(G.edges())]

            roots = [n for n, d in G.in_degree() if d == 0]
            root_node_id = roots[0]
            root_row = self.df[self.df["post_id"] == root_node_id]
            root_row = root_row.head(1)
            root_row["parent_id"] = 'nan'

            new_df = pd.concat([new_df, root_row])

            result = DelabTree(new_df, G)
            # is_valid = result.validate(verbose=True)
            # assert is_valid
            return result

    def as_attached_orphans(self, as_delab_tree=True):
        """
        In social media, many trees have missing posts. This can lead to a big loss of data.
        This algorithm attaches missing tweets to the root_post thus recreating a similar tree_structure
        :param as_delab_tree: if true returns a copy of the delab tree object, as nx graph else
        :return:
        """
        """
        if self.validate(verbose=False):
            if as_delab_tree:
                return self
            else:
                return self.reply_graph
        """

        roots = self.df[pd_is_nan(self.df["parent_id"])]
        missing_parents = get_missing_parents(self.df)

        # no clear roots but missing parents
        if len(roots.index) == 0:
            roots = self.df[self.df["parent_id"].isin(missing_parents)]

        if len(roots.index) > 1:
            # if there are more then one option for the root
            roots2 = roots[roots["post_id"] == roots["tree_id"]]
            if len(roots2.index) == 1:
                roots = roots2
            else:
                sorted_roots = roots.sort_values("created_at")
                roots = sorted_roots.head(1)

        # get the root node element from the root row
        roots = roots.reset_index(drop=True)
        try:
            root_node_id = roots.loc[0, ["post_id"]][0]
        except KeyError:
            self.validate(verbose=True)
            print("hello")

        # self.df.sort_values("created_at", inplace=True)

        def change_parents(parent_id):
            if parent_id in missing_parents or pd_is_nan(parent_id):
                return root_node_id
            else:
                return parent_id

        if as_delab_tree:
            df2 = self.df.copy()
            df2["parent_id"] = df2["parent_id"].apply(change_parents)
            df2.loc[df2["parent_id"] == df2["post_id"], "parent_id"] = 'nan'
            tree2 = DelabTree(df2)
            # tree2.reply_graph = tree2.as_tree() # no idea why
            # valid = tree2.validate(verbose=True)
            # assert valid
            return tree2
        else:
            self.df["parent_id"] = self.df["parent_id"].apply(change_parents)
            self.df.loc[self.df["parent_id"] == self.df["post_id"], "parent_id"] = 'nan'
            self.reply_graph = self.as_reply_graph()
            # self.reply_graph = self.as_tree()  # no idea why
            return self.reply_graph

    def as_merged_self_answers_graph(self, as_delab_tree=True, return_deleted=False):
        """
        subsequent posts of the same author are merged into one post
        :param as_delab_tree: if true, returns the DelabTree as a new object with the merged_self_answers
        :param return_deleted:
        :return:
        """
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
            # we need to assume that parent_id as a str can be converted to nan
            if pd_is_nan(parent_id):
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
                    assert moving_parent_id is not None and moving_parent_id != 'nan'
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
            if not (pd_is_nan(parent_id)):
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
        # print("removed {} and changed {}".format(to_delete_list, to_change_map))
        if as_delab_tree:
            self.reply_graph = G
            self.__update_df()
            return self
        if return_deleted:
            return G, to_delete_list, to_change_map
        return G

    def as_flow_duo(self, min_length_flows=6, min_post_branching=3, min_pre_branching=3, metric="sentiment",
                    verbose=False) -> FLowDuo:
        """
        compute the two flows of the tree that have the greatest difference regarding the metric
        :param min_length_flows:
        :param min_post_branching:
        :param min_pre_branching:
        :param metric: the dataframe needs to contain a column sentiment_value or toxicity_value,
        metric can be toxicity instead of default
        :param verbose:
        :return:
        """
        flows, longest = self.get_conversation_flows()

        candidate_flows: list[(str, list[DelabPost])] = []
        for name, tweets in flows.items():
            if len(tweets) < min_length_flows:
                continue
            else:
                candidate_flows.append((name, tweets))
        conversation_flow_duo_candidate, conversation_max = compute_highest_flow_delta(candidate_flows=candidate_flows,
                                                                                       metric=metric,
                                                                                       min_post_branching=
                                                                                       min_post_branching,
                                                                                       min_pre_branching=
                                                                                       min_pre_branching,
                                                                                       verbose=verbose)
        if conversation_flow_duo_candidate is None:
            return None
        name1 = conversation_flow_duo_candidate[0]
        name2 = conversation_flow_duo_candidate[1]
        flow_duo_result = FLowDuo(
            name1=name1,
            name2=name2,
            toxic_delta=conversation_max,
            posts1=flows[name1],
            posts2=flows[name2]
        )
        return flow_duo_result

    def get_conversation_flows(self) -> (dict[str, list[DelabPost]], str):
        """
        computes all flows (paths that lead from root to leaf) in the reply tree
        :rtype: object
        :return: flow_dict : str -> [DelabPost], name_of_longest : str
        """
        # reply_graph = self.as_reply_graph()
        root = get_root(self.reply_graph)
        leaves = [x for x in self.reply_graph.nodes() if self.reply_graph.out_degree(x) == 0]
        flows = []
        flow_dict = {}
        for leaf in leaves:
            paths = nx.all_simple_paths(self.reply_graph, root, leaf)
            flows.append(next(paths))
        for flow in flows:
            flow_name = str(flow[0]) + "_" + str(flow[-1])
            flow_tweets_frame = self.df[self.df[TABLE.COLUMNS.POST_ID].isin(flow)]
            flow_tweets = DelabPosts.from_pandas(flow_tweets_frame)
            flow_dict[flow_name] = flow_tweets

        name_of_longest = max(flow_dict, key=lambda x: len(set(flow_dict[x])))
        return flow_dict, name_of_longest

    def get_flow_candidates(self, length_flow: int, filter_function: Callable[[list[DelabPost]], bool] = None):
        flow_dict, name_longest = self.get_conversation_flows()
        # discarding the names of the flows

        # removing intersecting flows (preparation)
        post_ids_sets = {}
        for name, flow in flow_dict.items():
            post: DelabPost
            post_ids = set()
            for post in flow:
                post_ids.add(post.post_id)
            post_ids_sets[name] = post_ids
        # find intersecting flows
        flows_to_keep = set()
        removed_keys = set()
        for set_name1, post_id_set in post_ids_sets.items():
            for set_name2, post_id_set2 in post_ids_sets.items():
                if set_name1 in removed_keys or set_name2 in removed_keys:
                    continue
                if not post_id_set.isdisjoint(post_id_set2):
                    if len(post_id_set) > len(post_id_set2):
                        removed_keys.add(set_name2)
                        if set_name2 in flows_to_keep:
                            flows_to_keep.remove(set_name2)
                        flows_to_keep.add(set_name1)
                    else:
                        removed_keys.add(set_name1)
                        if set_name1 in flows_to_keep:
                            flows_to_keep.remove(set_name1)
                        flows_to_keep.add(set_name2)
        # update the dictionary to only keep the longest flows
        flow_dict_filtered = {key: value for key, value in flow_dict.items() if key in flows_to_keep}
        flows = flow_dict_filtered.values()

        if filter_function is not None:
            flows = list(filter(lambda x: filter_function(x), flows))

        # filter min_length_flow
        flows = list(filter(lambda x: len(x) > length_flow, flows))

        # filter_subsequent_authors
        # flows = list(filter(lambda x: len(set(list(map(lambda y: y.author_id, x)))) < length_flow, flows))

        return flows

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

    def get_average_author_metrics(self):
        """
        computes the same metrics as for specific authors but as the average of all authors in the graph
        :return:
        """
        # calculate the degree centrality of each node
        g = self.as_author_interaction_graph()
        closeness_centrality = statistics.mean(nx.closeness_centrality(g).values())
        katz_centrality = statistics.mean(nx.katz_centrality(g).values())
        betweenness_centrality = statistics.mean(nx.betweenness_centrality(g))
        baseline_author_vision = statistics.mean(self.get_baseline_author_vision())
        metric = AuthorMetric(closeness_centrality, betweenness_centrality, katz_centrality, baseline_author_vision)
        return metric

    def get_baseline_author_vision(self):
        author2baseline = {}
        author_interaction_graph, to_delete_list, to_change_map = self.as_merged_self_answers_graph(return_deleted=True,
                                                                                                    as_delab_tree=False)
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
            # assert 0 <= baseline <= 1
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

    def validate_internal_structure(self):
        is_dublicated = self.df[
            'post_id'].duplicated().any()
        assert not is_dublicated, "The 'post_id' column is not unique for tree with id {}.".format(
            self.conversation_id)
        df_and_graph_align = all(x in list(self.df["post_id"]) or x in list(self.df["parent_id"])
                                 for x in self.reply_graph.nodes())
        assert df_and_graph_align, "graph and dataframe out of sync"

    def __validate_cycles(self, verbose):
        try:
            cycles = nx.find_cycle(self.reply_graph)
            if len(cycles) > 0:
                if verbose:
                    print("the graph contains cycles: ", cycles)
                return False
        except NetworkXNoCycle:
            return True

    def __validate_orphans(self, verbose):
        if len(list(nx.weakly_connected_components(self.reply_graph))) > 1:
            roots = [n for n, d in self.reply_graph.in_degree() if d == 0]
            if verbose:
                print("number of roots are", len(roots))
            return False
        else:
            return True

    def __validate_node_names(self, verbose):
        nodes = self.reply_graph.nodes
        # check for all kinds of problems
        if 'NA' in nodes or 'nan' in nodes:
            if verbose:
                print("the na values should not be part of the tree nodes!")
            return False
        return True

    def __validate_multiple_edges(self, verbose):
        G = self.reply_graph
        # Check for multiple edges
        has_multiple_edges = False

        # Iterate over the edges
        for u, v in G.edges():
            if G.number_of_edges(u, v) > 1:
                has_multiple_edges = True
                break

        # Print the result
        if has_multiple_edges and verbose:
            print("The graph has multiple edges between nodes.")
        return not has_multiple_edges

    def __is_connected(self, verbose):

        result = nx.is_connected(self.reply_graph.to_undirected())
        if not result and verbose:
            print("it is not connected")
        return result

    def __validate_time_stamps_differ(self, verbose):
        created_at_set = set(self.df[TABLE.COLUMNS.CREATED_AT])
        result = len(created_at_set) == len(self.df.index)
        if verbose and not result:
            duplicates = self.df[self.df.duplicated(TABLE.COLUMNS.CREATED_AT)]
            print("all posts need to have a different time stamp:", duplicates)
        # assert len(created_at_set) == len(self.df.index), "all posts need to have a different time stamp!"
        return result

    def validate(self, verbose=True, check_for="all"):

        result = True
        # check for cycles only
        if check_for == 'cycles':
            result = self.__validate_cycles(verbose)
        else:
            if check_for == 'orphans':
                result = self.__validate_orphans(verbose)
            else:
                try:
                    roots = get_all_roots(self.reply_graph)
                    result = result and len(roots) == 1
                    if verbose and len(roots) != 1:
                        print("roots: ", roots)
                    result = result and self.__validate_node_names(verbose)
                    result = result and self.__validate_cycles(verbose)
                    result = result and self.__validate_orphans(verbose)
                    result = result and self.__is_connected(verbose)
                    result = result and self.__validate_multiple_edges(verbose)
                    result = result and self.__validate_time_stamps_differ(verbose)
                    result = result and nx.is_tree(self.reply_graph)
                    if verbose and not nx.is_tree(self.reply_graph):
                        print("is not a valid tree")
                except NetworkXPointlessConcept:
                    print("The graph is not a valid tree.")
                    result = False
            if verbose and not result:
                print("The graph with id {} is not a valid tree.".format(self.conversation_id))
                try:
                    self.paint_reply_graph()
                except Exception:
                    self.paint_faulty_graph()
        return result

    def paint_faulty_graph(self):
        # create a new dictionary of nodes with truncated labels
        new_labels = {node: convert_float_ids_to_readable_str(node)[-3:] for node in
                      self.reply_graph.nodes}
        # create a new figure with a larger canvas
        fig, ax = plt.subplots(figsize=(8, 6))
        # create a new graph with truncated labels
        G_truncated = nx.relabel_nodes(self.reply_graph, new_labels)
        pos = nx.spring_layout(G_truncated, k=1.5)
        nx.draw_networkx_nodes(G_truncated, pos=pos, node_size=600, node_color='blue')
        nx.draw_networkx_edges(G_truncated, pos=pos, width=2)
        nx.draw_networkx_labels(G_truncated, pos=pos, font_color='white')
        plt.show()

    def paint_reply_graph(self):
        tree = self.as_tree()

        # create a new dictionary of nodes with truncated labels
        new_labels = {node: convert_float_ids_to_readable_str(node)[-3:] for node in
                      self.reply_graph.nodes}
        # create a new figure with a larger canvas
        fig, ax = plt.subplots(figsize=(20, 8))
        # create a new graph with truncated labels
        tree = nx.relabel_nodes(tree, new_labels)

        pos = graphviz_layout(tree, prog="twopi")
        # add_attributes_to_plot(conversation_graph, pos, tree)
        nx.draw_networkx_labels(tree, pos)
        nx.draw(tree, pos)
        plt.show()

    def paint_author_graph(self):
        tree = self.as_author_graph()
        root = get_root(self.as_reply_graph())
        paint_bipartite_author_graph(tree, root)
