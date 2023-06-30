# functions for testing below
import os
import random
from datetime import datetime

from delab_trees import TreeManager
from delab_trees.delab_tree import DelabTree
from delab_trees.constants import TABLE

import pandas as pd

from delab_trees.recursive_tree.recursive_tree import TreeNode

"""
This contains a couple of test_trees to play around and test stuff
"""


def get_test_tree() -> DelabTree:
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
    # creates one tree
    test_tree = manager.random()
    return test_tree


def get_test_manager() -> TreeManager:
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

    # not connected
    d6 = d.copy()
    d6["tree_id"] = [6] * 4
    d6['parent_id'] = [None, 1, 42, 3]
    d6["author_id"] = ["james", "hannah", "jana", "john"]

    # contains cycle
    d7 = d.copy()
    d7["tree_id"] = [7] * 4
    d7['post_id'] = [1, 2, 3, 2]
    d7['parent_id'] = [None, 1, 2, 2]
    d7["author_id"] = ["james", "hannah", "jana", "john"]

    df1 = pd.DataFrame(data=d)
    df2 = pd.DataFrame(data=d2)
    df3 = pd.DataFrame(data=d3)
    df4 = pd.DataFrame(data=d4)
    df5 = pd.DataFrame(data=d5)
    df6 = pd.DataFrame(data=d6)
    df7 = pd.DataFrame(data=d7)

    # df_list = [df1, df2, df3, df4, df5, df6, df7]
    # TODO implement tree with cycles
    df_list = [df1, df2, df3, df4, df5, df6]
    df = pd.concat(df_list, ignore_index=True)
    manager = TreeManager(df)
    return manager


def get_social_media_trees(platform="twitter", n=None, context="production") -> TreeManager:
    assert platform == "twitter" or platform == "reddit", "platform needs to be reddit or twitter!"
    if context == "test":
        file = "../delab_trees/data/dataset_twitter_no_text.pkl"
        # file = "/home/dehne/PycharmProjects/delab/scriptspy/dataset_twitter_no_text.pkl"
    else:
        this_dir, this_filename = os.path.split(__file__)
        file = os.path.join(this_dir, 'data/dataset_twitter_no_text.pkl')
    file = file.replace("reddit", platform)
    df = pd.read_pickle(file)
    manager = TreeManager(df, n=n)
    return manager


def get_example_conversation_tree(lang="en"):
    twittery_en, twittery_de = __get_example_conversations()
    conversation_id = 1
    conv = twittery_en
    if lang == "de":
        conversation_id = 2
        conv = twittery_de

    root_text = conv[0]
    data = __generate_fake_tweet_data(root_text, 1, conversation_id, lang=lang)
    tree = TreeNode(data, data['post_id'], tree_id_name="tree_id")
    current_tree = tree
    index = 2
    for fake_tweet in conv[1:]:
        data = __generate_fake_tweet_data(fake_tweet, index, conversation_id, lang=lang)
        child = TreeNode(data, data['post_id'], current_tree.node_id, tree_id_name="tree_id")
        tree.find_parent_of(child)
        current_tree = child
        index += 1
    return tree


def get_sample_flow_test_tree():
    d = {'tree_id': [1] * 7,
         'post_id': [1, 2, 3, 4, 5, 6, 7],
         'parent_id': [None, 1, 2, 3, 3, 4, 6],
         'author_id': ["james", "mark", "steven", "john", "mark", "steven", "john"],
         'text': ["I am James", "I am Mark", " I am Steven", "I am John", "I am Mark", " I am Steven", "I am John"],
         "created_at": [pd.Timestamp('2017-01-01T01'),
                        pd.Timestamp('2017-01-01T02'),
                        pd.Timestamp('2017-01-01T03'),
                        pd.Timestamp('2017-01-01T04'),
                        pd.Timestamp('2017-01-01T05'),
                        pd.Timestamp('2017-01-01T06'),
                        pd.Timestamp('2017-01-01T07')
                        ]}
    df = pd.DataFrame(data=d)
    forest = TreeManager(df)
    return forest


def __generate_fake_tweet_data(text, id, conversation_id=1, lang="en"):
    data = {TABLE.COLUMNS.TEXT: text,
            TABLE.COLUMNS.POST_ID: id,
            TABLE.COLUMNS.AUTHOR_ID: 42,
            TABLE.COLUMNS.CREATED_AT: datetime.now(),
            TABLE.COLUMNS.TREE_ID: conversation_id,
            TABLE.COLUMNS.LANGUAGE: lang
            }
    return data


def __get_example_conversations():
    """
    this is freely invented
    :return:
    """
    tweet1 = '>>Warum darf man das wort Zigeuner-Soße eigentlich nicht mehr sagen?'
    ans1 = '>>weil das z-wort diskriminierent gegenüber Sinti und Roma ist'
    ans1_1 = '>>früher haben wir einfach gesagt was uns gefällt, da haben sich die scheiß Zigeuner doch auch nicht beschwert!'
    ans1_2 = '>>ernsthaft...?'
    ans1_3 = '>>bruder halts maul'
    ans2 = '>> man darf sagen was man will. In diesem Land gilt die Meinungfreiheit! diese linken Spasten wollen und den Mund verbieten!'
    ans2_1 = '>> isso!'
    ans2_2 = '>> heul doch.'
    ans3 = '>>frag ich mich auch! Als ob das irgendeinen Zigeuner juckt, die können doch eh kein deutsch'
    ans4 = '>> die scheiß zecken wollen doch nur dass wir spuren! lass dir nichts verbieten!'

    twitterconv_de = [tweet1, ans1, ans1_1, ans1_2, ans1_3, ans2, ans2_1, ans2_2, ans3, ans4]

    tweet1_en = '>>Happy International Day against Homophobia,Transphobia and Biphobia!'
    ans1_en = '>>wow! the fags are getting really creative with their days!'
    ans1_2_en = '>> for real!'
    ans1_3_en = '>> if you dont want to celebrate it, thats fine'
    ans2_en = '>> why are dykes always telling everyone they are dykes? i really dont care!'
    ans2_1_en = '>> when did they say they were a lesbian?:D'
    ans2_2_en = '>>i swear the lesbos are getting crazy'
    ans3_en = '>> yayy we love<3'
    ans4_en = '>> so everyday is homo day now, you think?'

    twitterconv_en = [tweet1_en, ans1_en, ans1_2_en, ans1_3_en, ans2_en, ans2_1_en, ans2_2_en, ans3_en, ans4_en]

    return twitterconv_en, twitterconv_de
