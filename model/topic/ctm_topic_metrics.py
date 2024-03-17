from typing import List

from topic.topic_metrics import TopicMetrics
from topic.topic_object import TopicObject


class CTMTopicMetrics(TopicMetrics):

    def __init__(self, number_of_topics, model, endpoint, word_number=10):
        super().__init__(number_of_topics, model, endpoint, word_number)

    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        topics: List[TopicObject] = []
        for idx in range(self.number_of_topics):
            top_words = self.model.get_topics()[idx]
            topics.append(TopicObject(id, top_words))
        return topics
