from typing import List

from delab_trees.constants import TABLE


class DelabPost:
    def __init__(self, post_id, parent_id, text: str, tree_id, author_id, created_at):
        self.post_id = post_id
        self.parent_id = parent_id
        self.text = text
        self.tree_id = tree_id
        self.author_id = author_id
        self.created_at = created_at


class DelabPosts:

    @staticmethod
    def from_pandas(df) -> List[DelabPost]:
        result = []
        for i in df.index:
            row = df.loc[i]
            post_id = row[TABLE.COLUMNS.POST_ID]
            parent_id = row[TABLE.COLUMNS.PARENT_ID]
            text = row[TABLE.COLUMNS.TEXT]
            tree_id = row[TABLE.COLUMNS.TREE_ID]
            author_id = row[TABLE.COLUMNS.AUTHOR_ID]
            created_at = row[TABLE.COLUMNS.CREATED_AT]
            post = DelabPost(post_id, parent_id, text, tree_id, author_id, created_at)
            result.append(post)
        return result
