from typing import List

from model.topic.topic_metrics import TopicMetrics
from model.topic.topic_object import TopicObject


class BertTopicMetrics(TopicMetrics):

    def __init__(self, number_of_topics, model, endpoint, word_number=10):
        super().__init__(number_of_topics, model, endpoint, word_number)

    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        topics: List[TopicObject] = []
        for i in range(self.number_of_topics):
            print(i)
            print([x[0] for x in self.model.get_topic(i)])
            top_words = [x[0] for x in self.model.get_topic(i)]
            topics.append(TopicObject(id, top_words))
        return topics