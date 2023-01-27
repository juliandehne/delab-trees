class AuthorMetric:
    def __init__(self, closeness_centrality, betweenness_centrality, katz_centrality, baseline_author_vision):
        self.closeness_centrality = closeness_centrality
        self.betweenness_centrality = betweenness_centrality
        self.katz_centrality = katz_centrality
        self.baseline_author_vision = baseline_author_vision


class AdvancedAuthorMetrics(AuthorMetric):
    def __init__(self, closeness_centrality, betweenness_centrality, katz_centrality, baseline_author_vision, rb_vision,
                 pb_vision):
        super(AuthorMetric, self).__init__(closeness_centrality,
                                           betweenness_centrality,
                                           katz_centrality,
                                           baseline_author_vision)
        self.rb_vision = rb_vision
        self.pb_vision = pb_vision
