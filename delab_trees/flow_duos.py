import logging

from delab_trees.delab_post import DelabPost

logger = logging.getLogger(__name__)

# settings concerning filtering flows
min_length_flows = 8
min_post_branching = 3
min_pre_branching = 3


class FLowDuo:
    """
    This class represents a couple of two Conversation Flows (reply chains from root to leaf) that
    have a big difference in given measures such as sentiment or toxicity
    """

    def __init__(self, name1, name2, toxic_delta, posts1, posts2):
        self.name1 = name1
        self.name2 = name2
        self.toxic_delta = toxic_delta
        self.posts1: list[DelabPost] = posts1
        self.posts2: list[DelabPost] = posts2


class FlowDuoWindow(FLowDuo):
    """
    This cuts a window in a duo flow dataset for inspection and qualitative analysis.
    """

    def __init__(self, name1, name2, toxic_delta, posts1, tweets2, post_branch_length, pre_branch_length):
        super().__init__(name1, name2, toxic_delta, posts1, tweets2)
        self.common_tweets: list[DelabPost] = []
        self.id2tweet = {}
        self.tweets1_post_branching: list[DelabPost] = []
        self.tweets2_post_branching: list[DelabPost] = []
        self.initialize_window(post_branch_length, pre_branch_length)

    def initialize_window(self, post_branch_length, pre_branch_length):
        self.posts1 = sorted(list(self.posts1), key=lambda x: x.created_at)
        self.posts2 = sorted(list(self.posts2), key=lambda x: x.created_at)
        flow_1_ids = []
        flow_2_ids = []
        # convert tweet lists to id lists
        for post in self.posts1:
            self.id2tweet[post.post_id] = post
            flow_1_ids.append(post.post_id)
        for post in self.posts2:
            self.id2tweet[post.post_id] = post
            flow_2_ids.append(post.post_id)
        intersection_ids = set(flow_1_ids).intersection(set(flow_2_ids))
        branching_index = max([flow_2_ids.index(intersect_id) for intersect_id in intersection_ids])
        intersection_id = flow_1_ids[branching_index]
        branching_index, intersection_id = compute_intersection_id_in_weird_trees(branching_index, flow_1_ids,
                                                                                  flow_2_ids, intersection_id)
        for post in self.posts1:
            if post.post_id != intersection_id:
                self.common_tweets.append(post)
            else:
                break
        # assert flow_2_ids.index(
        #    intersection_id) == branching_index, "the branching index should be the same for both branches"
        start_index_pre_branching = max(branching_index - pre_branch_length, 0)
        self.common_tweets = self.common_tweets[start_index_pre_branching:branching_index]
        end_index_post_branching = min(branching_index + 1 + post_branch_length, len(self.posts1), len(self.posts2))
        self.tweets1_post_branching = self.posts1[branching_index + 1: end_index_post_branching]
        self.tweets2_post_branching = self.posts2[branching_index + 1: end_index_post_branching]


def compute_highest_flow_delta(candidate_flows: list[(str, list[DelabPost])], metric: str, min_post_branching: int,
                               min_pre_branching: int, verbose: bool):
    conversation_delta_highscore = 0  # assert that there is only one flowduo candidate per conversation
    conversation_flow_duo_candidate = None
    for name, posts in candidate_flows:
        for name_2, posts_2 in candidate_flows:
            if name_2 != name:
                tweet_ids = set([tweet.post_id for tweet in posts])
                tweet2_ids = set([tweet.post_id for tweet in posts_2])
                n_pre_branching = len(tweet_ids.intersection(tweet2_ids))
                n_smaller_flow = min(len(tweet_ids), len(tweet2_ids))
                if n_pre_branching < min_pre_branching or (n_smaller_flow - n_pre_branching) < min_post_branching:
                    continue
                else:
                    pos_toxicity = 0
                    for positive_tweet in posts:
                        if metric == "toxicity":
                            if not hasattr(pos_toxicity, "toxic_value"):
                                raise Exception("INITIAL DATA needs column toxic_value!")
                            if positive_tweet.toxic_value is not None:
                                pos_toxicity += positive_tweet.toxic_value
                        else:
                            if not hasattr(pos_toxicity, "sentiment_value"):
                                raise Exception("INITIAL DATA needs column sentiment_value!")
                            if positive_tweet.sentiment_value is not None:
                                pos_toxicity += positive_tweet.sentiment_value
                    pos_toxicity = pos_toxicity / len(posts)

                    neg_toxicity = 0
                    for neg_tweet in posts_2:
                        if metric == "toxicity":
                            if neg_tweet.toxic_value is not None:
                                neg_toxicity += neg_tweet.toxic_value
                        else:
                            if neg_tweet.sentiment_value is not None:
                                neg_toxicity += neg_tweet.sentiment_value
                    neg_toxicity = neg_toxicity / len(posts_2)

                    if metric == "toxicity":
                        tox_delta = abs(pos_toxicity - neg_toxicity)
                    else:
                        tox_delta = 0
                        if (pos_toxicity <= 0 and neg_toxicity <= 0) or (pos_toxicity >= 0 and neg_toxicity >= 0):
                            tox_delta = abs(abs(pos_toxicity) - abs(neg_toxicity))
                        else:
                            if pos_toxicity >= 0 >= neg_toxicity:
                                tox_delta = pos_toxicity + abs(neg_toxicity)
                            if neg_toxicity >= 0 >= pos_toxicity:
                                tox_delta = neg_toxicity + abs(pos_toxicity)
                    # max_delta = max(max_delta, tox_delta)
                    if tox_delta > conversation_delta_highscore:
                        conversation_delta_highscore = tox_delta
                        conversation_flow_duo_candidate = (name, name_2)
    return conversation_flow_duo_candidate, conversation_delta_highscore


def flow_duos2flow_windows(dual_flows, post_branch_length=5, pre_branch_length=5):
    result = []
    for dual_flow in dual_flows:
        window = FlowDuoWindow(dual_flow.name1, dual_flow.name2, dual_flow.toxic_delta, dual_flow.posts1,
                               dual_flow.posts2, post_branch_length, pre_branch_length)
        result.append(window)
    return result


def compute_intersection_id_in_weird_trees(branching_index, flow_1_ids, flow_2_ids, intersection_id):
    # this computes the duo_flow common ids in case of a mention (non-deterministic tree structure)
    if intersection_id not in flow_2_ids or (flow_2_ids.index(intersection_id) != branching_index):
        branching_index = 0
        for id in flow_1_ids:
            for id2 in flow_2_ids:
                if id == id2:
                    branching_index += 1
                else:
                    break
    intersection_id = flow_1_ids[branching_index]
    return branching_index, intersection_id


def compute_flow_name(flow: list[DelabPost], prefix=""):
    first = flow[0]
    last = flow[-1]
    return prefix + str(first.post_id) + "_" + str(last.post_id)
