from typing import List

from model.topic.topic_metrics import TopicMetrics
from model.topic.topic_object import TopicObject


class STMCudaTopicMetrics(TopicMetrics):

    def __init__(self, number_of_topics, model, endpoint, word_number=10):
        super().__init__(number_of_topics, model, endpoint, word_number)

    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        topics: List[TopicObject] = []
        for id, topic in enumerate(self.model):
            top = TopicObject(id, topic)
            topics.append(top)
        return topics
