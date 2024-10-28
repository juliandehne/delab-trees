# Delab Trees

A library to analyze conversation trees. 

## Installation 

pip install delab_trees

## Learning Resource

In order to learn more about the library including some metrics and research questions that can be answered, have a look at the more detailed [jupyter notebook](index.ipynb). It uses the [Quarto Jupyterlab Extension](https://quarto.org/docs/tools/jupyter-lab-extension.html) for rendering. Use this for a better readability. 

```python
python3 -m pip install jupyterlab-quarto
```


## Get started

Example data for Reddit and Twitter are available here https://github.com/juliandehne/delab-trees/raw/main/delab_trees/data/dataset_[reddit|twitter]_no_text.pkl. 
The data is structure only. Ids, text, links, or other information that would break confidentiality of the academic 
access have been omitted.


The trees are loaded from tables like this:

|    |   tree_id |   post_id |   parent_id | author_id   | text        | created_at          |
|---:|----------:|----------:|------------:|:------------|:------------|:--------------------|
|  0 |         1 |         1 |         nan | james       | I am James  | 2017-01-01 01:00:00 |
|  1 |         1 |         2 |           1 | mark        | I am Mark   | 2017-01-01 02:00:00 |
|  2 |         1 |         3 |           2 | steven      | I am Steven | 2017-01-01 03:00:00 |
|  3 |         1 |         4 |           1 | john        | I am John   | 2017-01-01 04:00:00 |
|  4 |         2 |         1 |         nan | james       | I am James  | 2017-01-01 01:00:00 |
|  5 |         2 |         2 |           1 | mark        | I am Mark   | 2017-01-01 02:00:00 |
|  6 |         2 |         3 |           2 | steven      | I am Steven | 2017-01-01 03:00:00 |
|  7 |         2 |         4 |           3 | john        | I am John   | 2017-01-01 04:00:00 |

This dataset contains two conversational trees with four posts each.

Currently, you need to import conversational tables as a pandas dataframe like this:

## Explanation

The `TreeManager` object in `delab_trees` holds a dictionary of trees. The keys in this dictionary are `tree_id`s, and the values are the actual tree structures, each represented as a `DelabTree` object. This setup allows you to access individual conversation trees by their unique ID, enabling easy retrieval and manipulation of specific trees.

Each `DelabTree` instance contains two main attributes:

- **`self.reply_graph`**: a NetworkX `DiGraph` representation of the conversation as a directed graph.
- **`self.df`**: a pandas `DataFrame` representation of the conversation as a table, preserving the structure and metadata of each post.

### Example

To demonstrate, let’s create a sample dataset and initialize the `TreeManager`. We'll then access an individual tree using its ID and show both representations (graph and table) of that tree.

```python
import pandas as pd
from delab_trees import TreeManager

# Sample dataset with two separate conversational trees (tree_id 1 and 2)
data = {
    'tree_id': [1, 1, 1, 1, 2, 2, 2, 2],
    'post_id': [1, 2, 3, 4, 1, 2, 3, 4],
    'parent_id': [None, 1, 2, 1, None, 1, 2, 3],
    'author_id': ["james", "mark", "steven", "john", "james", "mark", "steven", "john"],
    'text': ["I am James", "I am Mark", "I am Steven", "I am John",
             "I am James again", "I am Mark again", "I am Steven again", "I am John again"],
    'created_at': [pd.Timestamp('2017-01-01T01'),
                   pd.Timestamp('2017-01-01T02'),
                   pd.Timestamp('2017-01-01T03'),
                   pd.Timestamp('2017-01-01T04'),
                   pd.Timestamp('2018-01-01T01'),
                   pd.Timestamp('2018-01-01T02'),
                   pd.Timestamp('2018-01-01T03'),
                   pd.Timestamp('2018-01-01T04')]
}

# Load data into a DataFrame
df = pd.DataFrame(data)

# Initialize TreeManager with the DataFrame
manager = TreeManager(df)

# Access the dictionary of trees
trees = manager.trees

# Verify the structure of the dictionary (keys and types of values)
print("Keys in TreeManager dictionary (tree IDs):", trees.keys())  # Output: dict_keys([1, 2])

# Access a specific tree by tree_id
tree_id = 1
tree = trees[tree_id]

# Display the NetworkX graph representation of the tree
print("Graph representation of tree:", tree.reply_graph)

# Display the DataFrame (table) representation of the tree
print("Table representation of tree:\n", tree.df)
```

You can now analyze the reply trees basic metrics:

```python
from delab_trees.main import get_test_tree
from delab_trees.delab_tree import DelabTree

test_tree : DelabTree = get_test_tree()
assert test_tree.total_number_of_posts() == 4
assert test_tree.average_branching_factor() > 0
```

A summary of basic metrics can be attained by calling

```python
from delab_trees.test_data_manager import get_test_tree
from delab_trees.delab_tree import DelabTree

test_tree : DelabTree = get_test_tree()
print(test_tree.get_author_metrics())

# >>> removed [] and changed {} (merging subsequent posts of the same author)
# >>>{'james': <delab_trees.delab_author_metric.AuthorMetric object at 0x7fa9c5496110>, 'steven': <delab_trees.delab_author_metric.AuthorMetric object at 0x7fa9c5497dc0>, 'john': <delab_trees.delab_author_metric.AuthorMetric object at 0x7fa9c5497a00>, 'mark': <delab_trees.delab_author_metric.AuthorMetric object at 0x7fa9c5497bb0>}



```

## Library Functions

| Function Name                  | Parameters                                                                                                                                     | Return Type                              | Description                                                                                |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------|
| `__init__`                     | `df: pd.DataFrame`, `g: MultiDiGraph = None`                                                                                                   | `None`                                   | Initializes a `DelabTree` instance with a pandas DataFrame and an optional graph.          |
| `__str__`                      | `None`                                                                                                                                         | `str`                                    | Returns a string representation of the `DelabTree`.                                        |
| `from_recursive_tree`          | `root_node: TreeNode`                                                                                                                          | `DelabTree`                              | Creates a `DelabTree` instance from a `TreeNode` recursive structure.                      |
| `branching_weight`             | `None`                                                                                                                                         | `float`                                  | Calculates the branching weight of the tree.                                               |
| `average_branching_factor`     | `None`                                                                                                                                         | `float`                                  | Computes the average branching factor of the tree.                                         |
| `root_dominance`               | `None`                                                                                                                                         | `float`                                  | Computes a dominance metric for the root author in the conversation.                       |
| `total_number_of_posts`        | `None`                                                                                                                                         | `int`                                    | Returns the total number of posts in the conversation.                                     |
| `depth`                        | `None`                                                                                                                                         | `int`                                    | Calculates the depth of the reply graph.                                                   |
| `as_reply_graph`               | `None`                                                                                                                                         | `DiGraph`                                | Generates a directed reply graph from the DataFrame.                                       |
| `as_author_graph`              | `None`                                                                                                                                         | `MultiDiGraph`                           | Creates a directed graph combining reply and author relations.                             |
| `as_author_interaction_graph`  | `None`                                                                                                                                         | `DiGraph`                                | Projects the author interaction graph to capture who answered whom.                        |
| `as_tree`                      | `None`                                                                                                                                         | `DiGraph`                                | Creates a tree representation of the reply graph using BFS traversal.                      |
| `as_post_list`                 | `None`                                                                                                                                         | `list[DelabPost]`                        | Converts the DataFrame to a list of `DelabPost` objects.                                   |
| `as_recursive_tree`            | `None`                                                                                                                                         | `TreeNode`                               | Converts the DataFrame into a recursive tree structure.                                    |
| `as_biggest_connected_tree`    | `stateless: bool = True`                                                                                                                       | `Union[DelabTree, Graph]`                | Finds and returns the largest connected component of the reply graph.                      |
| `as_removed_cycles`            | `as_delab_tree: bool = True`, `compute_arborescence: bool = False`                                                                             | `Union[DiGraph, DelabTree]`              | Removes cycles in the reply graph and optionally computes a minimum spanning arborescence. |
| `as_attached_orphans`          | `as_delab_tree: bool = True`                                                                                                                   | `Union[DelabTree, DiGraph]`              | Attaches orphaned nodes to the root node to recreate a similar tree structure.             |
| `as_merged_self_answers_graph` | `as_delab_tree: bool = True`, `return_deleted: bool = False`                                                                                   | `Union[DiGraph, DelabTree, tuple]`       | Merges sequential posts by the same author into one post, returning a new `DelabTree`.     |
| `as_flow_duo`                  | `min_length_flows: int = 6`, `min_post_branching: int = 3`, `min_pre_branching: int = 3`, `metric: str = "sentiment"`, `verbose: bool = False` | `FLowDuo`                                | Computes the two conversation flows with the highest difference in a specified metric.     |
| `get_conversation_flows`       | `as_list: bool = False`                                                                                                                        | `tuple[dict[str, list[DelabPost]], str]` | Returns all conversation flows (paths from root to leaf).                                  |
| `get_flow_candidates`          | `length_flow: int`, `filter_function: Callable[[list[DelabPost]], bool] = None`                                                                | `list[list[DelabPost]]`                  | Filters conversation flows by length and an optional filter function.                      |
| `get_author_metrics`           | `None`                                                                                                                                         | `dict[str, AuthorMetric]`                | Computes centrality metrics (closeness, betweenness, Katz) for each author.                |
| `get_average_author_metrics`   | `None`                                                                                                                                         | `AuthorMetric`                           | Calculates average centrality metrics for all authors in the graph.                        |
| `get_baseline_author_vision`   | `None`                                                                                                                                         | `dict[str, float]`                       | Computes a baseline vision score for each author based on reply behavior.                  |
| `get_single_author_metrics`    | `author_id: str`                                                                                                                               | `AuthorMetric`                           | Returns centrality metrics for a specific author.                                          |
| `validate_internal_structure`  | `None`                                                                                                                                         | `None`                                   | Validates that the DataFrame has unique post IDs and aligns with the graph structure.      |
| `validate`                     | `verbose: bool = True`, `check_for: str = "all"`, `check_time_stamps_differ: bool = True`                                                      | `bool`                                   | Validates the graph structure, including cycles, connectivity, and node names.             |
| `paint_faulty_graph`           | `None`                                                                                                                                         | `None`                                   | Visualizes a faulty reply graph with truncated node labels.                                |
| `paint_reply_graph`            | `None`                                                                                                                                         | `None`                                   | Draws the reply graph in a circular layout.                                                |
| `paint_author_graph`           | `None`                                                                                                                                         | `None`                                   | Visualizes the author interaction graph.                                                   |



## Advanced Use Cases 

More complex metrics that use the full dataset for training can be gotten by the manager:

```python
import pandas as pd
from delab_trees import TreeManager

d = {'tree_id': [1] * 4,
     'post_id': [1, 2, 3, 4],
     'parent_id': [None, 1, 2, 1],
     'author_id': ["james", "mark", "steven", "john"],
     'text': ["I am James", "I am Mark", " I am Steven", "I am John"],
     "created_at": [pd.Timestamp('2017-01-01T01'),
                    pd.Timestamp('2017-01-01T02'),
                    pd.Timestamp('2017-01-01T03'),
                    pd.Timestamp('2017-01-01T04')]}
df = pd.DataFrame(data=d)
manager = TreeManager(df) # creates one tree
rb_vision_dictionary : dict["tree_id", dict["author_id", "vision_metric"]] = manager.get_rb_vision()
```

The following two complex metrics are implemented: 

```python
from delab_trees.test_data_manager import get_test_manager

manager = get_test_manager()
rb_vision_dictionary = manager.get_rb_vision() # predict an author having seen a post
pb_vision_dictionary = manager.get_pb_vision() # predict an author to write the next post
```

## How to cite

```latex
    @article{dehne_dtrees_23,
    author    = {Dehne, Julian},
    title     = {Delab-Trees: measuring deliberation in online conversations},        
    url = {https://github.com/juliandehne/delab-trees}     
    year      = {2023},
}

```
