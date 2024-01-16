import os
import pickle
from typing import List

from model.preprocessing.dataset_interface import DatasetInterface
from model.preprocessing.lemma_tokenizer import LemmaTokenizer


class DataSetFullImpl(DatasetInterface):

    @staticmethod
    def load(folder, name):
        path = os.path.join(folder, name)
        modelObject = pickle.load(open(path, "rb"))
        return modelObject

    def save(self, folder_name, name=None):
        results_path = os.path.join(folder_name, name)
        with open(results_path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def test_labels(self) -> List:
        return self._test_labels

    def train_labels(self) -> List:
        return self._train_labels

    def train_tokens(self) -> List:
        return self._doc_tokens_training

    def test_tokens(self) -> List:
        return self._doc_tokens_test

    def features(self) -> List:
        return self._features

    def frequency_map(self) -> dict:
        return self._freq_map

    def categories(self) -> list:
        return self._categories

    def __init__(self, feature_limit: int):

        self._feature_limit = feature_limit
        self.tokenizer = LemmaTokenizer()
        self._features = None
        self._freq_map = None
        self._doc_tokens_training = None
        self._doc_tokens_test = None
        self._test_labels = None
        self._train_labels = None
        self._categories = None

    def document_tokens(self):
        return self._doc_tokens_training

    def preprocess_data_set(self, docs_training, docs_test, train_labels, test_labels, categories):
        self._categories = categories
        self._train_labels = train_labels + test_labels
        self._test_labels = train_labels + test_labels
        self._doc_tokens_training = self.tokenizer.tokenize(docs_training + docs_test)
        self.feature_selection()
        self._doc_tokens_training, self._train_labels = self.reduce_tokens(self._doc_tokens_training,
                                                                           self._train_labels)
        self._doc_tokens_test = self._doc_tokens_training
        self._test_labels = self.train_labels()

    def feature_selection(self):
        freq_map, freq_doc_map = self.calculate_tokens_freq(self._doc_tokens_training)
        true_features = [(f, freq_map[f]) for f in freq_map.keys()]
        true_features = list(sorted(true_features, key=lambda x: x[1], reverse=True))[:self._feature_limit]
        true_features_list = list(sorted([pair[0] for pair in true_features]))
        self._features = true_features_list
        self._freq_map = {word: frq for (word, frq) in freq_doc_map.items() if word in true_features_list}

    def calculate_tokens_freq(self, tokenize_docs: list) -> (dict, dict):
        feature_frequency = {}  # used for feature selection
        feature_doc_occurence = {}  # used for probability Function
        for doc in tokenize_docs:
            doc_set = set()
            for token in doc:
                if token in feature_frequency:
                    feature_frequency[token] += 1
                else:
                    feature_frequency[token] = 1
                if token not in doc_set:
                    feature_doc_occurence[token] = 1 if token not in feature_doc_occurence else feature_doc_occurence[
                                                                                                    token] + 1
                doc_set.add(token)
        return feature_frequency, feature_doc_occurence

    def reduce_tokens(self, tokens, labels):
        filtered_tokens = []
        filtered_labels = []
        for id, doc_tokens in enumerate(tokens):
            tokens_tmp = [w.lower() for w in doc_tokens if w in self._features]
            if len(tokens_tmp) == 0:
                continue
            filtered_tokens.append(tokens_tmp)
            filtered_labels.append(labels[id])
        return filtered_tokens, filtered_labels
