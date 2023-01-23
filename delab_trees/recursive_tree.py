import logging

from delab_trees.delab_tree import TABLE
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class TreeNode:
    def __init__(self, data, post_id: int, parent=None, parent_type="replied_to"):
        self.post_id = post_id
        self.data = data
        self.children = []
        self.max_path_length = 0
        self.parent = parent
        self.parent_type = parent_type

    def find_parent_of(self, node):
        if type(self.post_id) is not type(node.parent.post_id):
            self.post_id = int(self.post_id)
            node.parent.post_id = int(node.parent.post_id)
        assert type(self.post_id) is type(node.parent.post_id)
        if node.parent.post_id == self.post_id:
            self.children.append(node)
            return True
        else:
            for child in self.children:
                result = child.find_parent_of(node)
                if result:
                    return result
        return False

    def print_tree(self, level):
        """level 0 is the root node, then incremented for subsequent generations"""
        print(f'{level * "_"}{level}: {self.data}')
        level += 1
        for child in self.children:
            child.print_tree(level)

    def to_string(self, level=0):
        result = ""
        if level == 0:
            result += "Conversation: " + str(self.data[TABLE.COLUMNS.TREE_ID]) + "\n\n"
        result += (level * "\t") + self.data_to_string(level)
        for child in self.children:
            result += child.to_string(level + 1)
        return result

    def data_to_string(self, level):
        text = self.data[TABLE.COLUMNS.TEXT].split(".")
        tabbed_text = []
        for sentence in text:
            sentence = sentence.replace('\n', ' ').replace('\r', '')
            tabbed_sentence = ""
            if len(sentence) > 125:
                tabbed_sentence += sentence[0:125]
                tabbed_sentence += "\n" + (level * "\t")
                tabbed_sentence += sentence[125:]
            else:
                tabbed_sentence = sentence
            tabbed_text.append(tabbed_sentence)
        separator = ".\n" + (level * "\t")
        tabbed_text = "\n" + (level * "\t") + separator.join(tabbed_text)
        return str(self.data.get("tw_author__name", "namenotgiven")) + "/" + str(
            self.data.get("tw_author__location", "locationnotgiven")) + "/" + str(
            self.data[TABLE.COLUMNS.AUTHOR_ID]) + ":" + tabbed_text + "\n\n"

    def to_norm_xml(self, level=0):
        discourse_elem = ET.Element('discourse')
        platform_elem = ET.SubElement(discourse_elem, 'platform')
        platform_elem.text = self.data['platform']
        speech_acts_elem = ET.SubElement(discourse_elem, 'speech-acts')
        self.tweet_to_speech_act_xml(speech_acts_elem)
        return ET.tostring(discourse_elem, encoding='utf-8')

    def tweet_to_speech_act_xml(self, parent_elem):
        speech_act_elem = ET.SubElement(parent_elem, 'speech-act')
        speech_act_id_elem = ET.SubElement(speech_act_elem, 'speech-act-id')
        speech_act_id_elem.text = str(self.post_id)
        author_elem = ET.SubElement(speech_act_elem, 'author')
        author_id_elem = ET.SubElement(author_elem, 'author-id')
        author_id_elem.text = str(self.data["author_id"])
        author_name_elem = ET.SubElement(author_elem, 'author-name')
        author_name_elem.text = self.data["tw_author__name"]
        text_elem = ET.SubElement(speech_act_elem, 'text')
        text_elem.text = self.data["text"]
        in_response_elem = ET.SubElement(speech_act_elem, 'in-reply-to')
        in_response_elem.text = str(self.data["tn_parent"])
        for child in self.children:
            child.tweet_to_speech_act_xml(speech_act_elem)

    def flat_size(self):
        children_size = 0
        for child in self.children:
            children_size += child.flat_size()
        return 1 + children_size

    def compute_max_path_length(self, level=0):
        # print(level)
        if len(self.children) > 0:
            child_max_paths = []
            for child in self.children:
                # print(["child"]*level)
                child_max_paths.append(child.compute_max_path_length(level + 1))
            return max(child_max_paths)
        return level

    def get_max_path_length(self):
        if self.max_path_length == 0:
            self.max_path_length = self.compute_max_path_length()
        return self.max_path_length

    def crop_orphans(self, max_orphan_count=4):
        favourite_children = []
        if len(self.children) > 0:
            counter = 0
            for child in self.children:
                if len(child.children) > 0 and counter < max_orphan_count:
                    favourite_children.append(child)
                    counter += 1
                    child.crop_orphans(max_orphan_count)
        self.children = favourite_children

    def all_tweet_ids(self):
        result = [self.post_id]
        for child in self.children:
            result.append(child.post_id)
        return result
