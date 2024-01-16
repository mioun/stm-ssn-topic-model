from model.topic.bert_topic_metrics import BertTopicMetrics
from model.topic.btm_topic_metrics import BTMTopicMetrics
from model.topic.lda_topic_metrics import LdaTopicMetrics
from model.topic.stm_topic_metrics import STMTopicMetrics
from model.topic.topic_metrics import TopicMetrics


class TopicMetricsFactory:

    @staticmethod
    def get_metric(model_type, number_of_topics, model, endpoint, word_number=10) -> TopicMetrics:
        if 'STM' == model_type:
            return STMTopicMetrics(number_of_topics, model, endpoint, word_number)

        if 'LDA' == model_type:
            return LdaTopicMetrics(number_of_topics, model, endpoint, word_number)

        if 'BTM' == model_type:
            return BTMTopicMetrics(number_of_topics, model, endpoint, word_number)

        raise Exception(f'Topic metrics not implemented for {model_type}')
