import string


class TopicObject():

    def __init__(self, id, words):
        self.metrics = {}
        self.words = words
        self.weights = {}
        self.id = id

    def set_coherence_metrics(self, metrics):
        self.metrics = metrics

    def get_metric(self, metric: string) -> float:
        return self.metrics[metric]

    def set_metric(self, metric: string, value: float):
        self.metrics[metric] = value

    def to_string(self):
        return self.words
