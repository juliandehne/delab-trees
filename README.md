# Delab Trees

A library to analyze conversation trees. 

## Installation 

pip install delab_trees

## Get started

Example data for Reddit and Twitter are available here https://github.com/juliandehne/delab-trees/raw/main/delab_trees/data/dataset_[reddit|twitter]_no_text.pkl. 

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
manager = TreeManager(df) 
manager.initialize_trees() # creates one tree
test_tree = manager.random()
```

Note that the tree structure is based on the parent_id matching another rows post_id. 

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
from delab_trees.main import get_test_tree
from delab_trees.delab_tree import DelabTree

test_tree : DelabTree = get_test_tree()
print(test_tree.get_author_metrics())

# >>> removed [] and changed {} (merging subsequent posts of the same author)
# >>>{'james': <delab_trees.delab_author_metric.AuthorMetric object at 0x7fa9c5496110>, 'steven': <delab_trees.delab_author_metric.AuthorMetric object at 0x7fa9c5497dc0>, 'john': <delab_trees.delab_author_metric.AuthorMetric object at 0x7fa9c5497a00>, 'mark': <delab_trees.delab_author_metric.AuthorMetric object at 0x7fa9c5497bb0>}

```

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
manager = TreeManager(df) 
manager.initialize_trees() # creates one tree
rb_vision_dictionary : dict["tree_id", dict["author_id", "vision_metric"]] = manager.get_rb_vision()
```

The following two complex metrics are implemented: 

```python
from delab_trees.main import get_test_manager

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
