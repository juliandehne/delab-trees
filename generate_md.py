import pandas as pd

d = {'tree_id': [1] * 4,
     'post_id': [1, 2, 3, 4],
     'parent_id': [None, 1, 2, 1],
     'author_id': ["james", "mark", "steven", "john"],
     'text': ["I am James", "I am Mark", " I am Steven", "I am John"],
     "created_at": [pd.Timestamp('2017-01-01T01'),
                    pd.Timestamp('2017-01-01T02'),
                    pd.Timestamp('2017-01-01T03'),
                    pd.Timestamp('2017-01-01T04')]}
d2 = d.copy()
d2["tree_id"] = [2] * 4
d2['parent_id'] = [None, 1, 2, 3]
d3 = d.copy()
d3["tree_id"] = [3] * 4
d3['parent_id'] = [None, 1, 1, 1]
# a case where an author answers himself
d4 = d.copy()
d4["tree_id"] = [4] * 4
d4["author_id"] = ["james", "james", "james", "john"]

d5 = d.copy()
d5["tree_id"] = [5] * 4
d5['parent_id'] = [None, 1, 2, 3]
d5["author_id"] = ["james", "james", "james", "john"]

df1 = pd.DataFrame(data=d)
df2 = pd.DataFrame(data=d2)
df3 = pd.DataFrame(data=d3)
df4 = pd.DataFrame(data=d4)
df5 = pd.DataFrame(data=d5)
df_list = [df1, df2, df3, df4, df5]

df = pd.concat(df_list, ignore_index=True)
print(df.to_markdown())
