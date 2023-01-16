class GraphNotInitializedException(Exception):
    def __init__(self, message="You need to call the initialize method before using metrics"):
        self.message = message
        super().__init__(self.message)
