from delab_tree import GRAPH, TABLE


class DelabPost:
    def __init__(self, post_id, parent_id, text: str):
        self.post_id = post_id
        self.parent_id = parent_id
        self.text = text


class DelabPosts:

    @staticmethod
    def from_pandas(df):
        result = []
        for i in df.index:
            row = df.loc[i]
            post_id = row[TABLE.COLUMNS.POST_ID]
            parent_id = row[TABLE.COLUMNS.PARENT_ID]
            text = row[TABLE.COLUMNS.TEXT]
            post = DelabPost(post_id, parent_id, text)
            result.append(post)
        return result
