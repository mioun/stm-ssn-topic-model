from typing import List

from model.network.stm_model_runner import STMModelRunner
from model.topic.topic_metrics import TopicMetrics
from model.topic.topic_object import TopicObject


class STMTopicMetrics(TopicMetrics):

    def __init__(self, number_of_topics, model: STMModelRunner, endpoint, word_number=15):
        super().__init__(number_of_topics, model, endpoint, word_number)

    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        return self.model.extract_topics_from_model(word_number)
