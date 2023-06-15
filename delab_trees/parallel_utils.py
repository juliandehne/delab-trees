import multiprocessing

from delab_trees.delab_tree import DelabTree


def compute_optimal_cpu_count():
    result = max(multiprocessing.cpu_count() - 2, 1)
    return result


# functions for parallel below

def create_trees_from_grouped(n_trees, groupings):
    trees = {}
    counter = 0
    for tree_id, df in groupings.items():
        counter += 1
        if n_trees is not None and counter > n_trees:
            break
        tree = DelabTree(df)
        trees[tree_id] = tree
    return trees


def create_trees_from_grouped_helper(groupings):
    return create_trees_from_grouped(None, groupings)


def compute_tre_map_f(tree_map_f, tree_items_group):
    key = tree_items_group[0]
    tree = tree_items_group[1]
    new_tree_local: DelabTree = getattr(tree, tree_map_f)(tree)

    return new_tree_local.df, {key: new_tree_local}
