import csv
import os
import pickle
import string
from abc import ABC, abstractmethod

import requests
from typing import List
import time

from model.topic.topic_object import TopicObject


class TopicMetrics(ABC):

    def __init__(self, number_of_topics, model, endpoint, word_number):
        self.number_of_topics = number_of_topics
        self.model = model
        self.coherence_metrics = ['npmi', 'ca']
        self.endpoint = endpoint
        self.topics: List[TopicObject] = self.extract_topics_from_model(word_number)

    @abstractmethod
    def extract_topics_from_model(self, word_number=10) -> List[TopicObject]:
        pass

    def palmetto_request(self, metric, topic_words: []):
        response = requests.get('{endpint}{metric}?words={tokens}'.format(endpint=self.endpoint, metric=metric,
                                                                          tokens='%20'.join(topic_words)))
        while response.status_code != 200:
            response = requests.get('{endpint}{metric}?words={tokens}'.format(endpint=self.endpoint, metric=metric,
                                                                              tokens='%20'.join(topic_words)))
            time.sleep(5)
            print("retrying")

        return float(response.text)

    def generate_metrics(self):
        unique = set()
        for topis in self.topics:
            unique.update(topis.words)
            print(topis.words)
        print(len(unique) / (len(self.topics) * 10))
        for topic in self.topics:
            print("Requestign metrics  for {topic_id}".format(topic_id=topic.words))
            for metric in self.coherence_metrics:
                metric_val = self.palmetto_request(metric, topic_words=topic.words)
                topic.set_metric(metric, metric_val)

    def save_results_csv(self, data_set, topic_number, model_name, output_path):
        results = []
        for metric in self.coherence_metrics:
            results.append([data_set, topic_number, model_name, metric,
                            round(self.get_average_metric_for_top_n(metric), 3)])
        results.append([data_set, topic_number, model_name, 'PUW', round(self.calualte_uniqe(), 3)])
        coherence_results_csv = open(os.path.join(output_path, '_coherence-results.csv'), 'a+')
        writer = csv.writer(coherence_results_csv, dialect='unix')
        for line in results:
            print(line)
            writer.writerow(line)
        coherence_results_csv.flush()

    def save(self, folder, name):
        results_path = os.path.join(folder, name)
        with open(results_path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(folder, name):
        path = os.path.join(folder, name)
        modelObject = pickle.load(open(path, "rb"))
        return modelObject

    def get_average_metric_for_top_n(self, metric, n=100.0) -> float:
        topics_for_evaluation = self.get_top_n(metric, n)
        avg = round(sum(map(lambda t: t.get_metric(metric), topics_for_evaluation)) / len(topics_for_evaluation), 5)
        return avg

    def get_top_n(self, metric: string, n=100.0) -> List[TopicObject]:
        topic_number = int((n / 100) * self.number_of_topics)
        topics_for_extraction: List[TopicObject] = sorted(self.topics, key=lambda x: x.get_metric(metric), reverse=True)
        return topics_for_extraction[:topic_number]

    def calualte_uniqe(self):
        uniq = set()
        for t in self.topics:
            uniq.update(t.words)
        return len(uniq) / (len(self.topics) * 10)
