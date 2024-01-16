from typing import List

import numpy as np

from model.topic.topic_metrics import TopicMetrics
from model.topic.topic_object import TopicObject


class BTMTopicMetrics(TopicMetrics):

    def __init__(self, number_of_topics, model, endpoint, word_number=10):
        super().__init__(number_of_topics, model, endpoint, word_number)

    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        topics: List[TopicObject] = []
        for id,row in enumerate(self.model.matrix_topics_words_):
            top_words = [self.model.vocabulary_[id] for id in np.argsort(row)[-word_number:][::-1]]
            topics.append(TopicObject(id, top_words))
        return topics
