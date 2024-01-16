from typing import List
from model.topic.topic_metrics import TopicMetrics
from model.topic.topic_object import TopicObject


class LdaTopicMetrics(TopicMetrics):

    def __init__(self, number_of_topics, model, endpoint, word_number=10):
        super().__init__(number_of_topics, model, endpoint, word_number)

    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        id2word = self.model.id2word
        topics: List[TopicObject] = []
        for id, topic in enumerate(self.model.get_topics()):
            word_weight_pairs = [(id2word.id2token[idx], val) for idx, val in enumerate(topic)]
            word_weight_pairs = sorted(word_weight_pairs, key=lambda x: x[1], reverse=True)
            top_words = [pair[0] for pair in word_weight_pairs[:word_number]]
            top = TopicObject(id, top_words)
            top.weights = {p[0]: p[1] for p in word_weight_pairs[:word_number]}
            topics.append(top)
            print(top_words)
        return topics

