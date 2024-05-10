import math
import os
import random
from abc import ABC
from math import exp

import torch

from model.preprocessing.dataset_interface import DatasetInterface


class CUDADataset(ABC):

    def __init__(self, data_path,
                 data_set: DatasetInterface,
                 data_set_name,
                 batch_size,
                 alpha,
                 epochs=1,
                 number_of_steps: int = 150,
                 squeeze: int = 3,
                 seed=666):

        self.squeeze = squeeze
        self.seed = seed
        self.data_set_name = data_set_name
        self.number_of_steps = number_of_steps
        self.alpha = alpha
        self.epochs = epochs
        self.data_path = data_path
        self.batch_size = batch_size
        self.data_set = data_set
        self.number_of_batches = math.ceil(len(self.data_set.train_tokens()) / batch_size)
        self.labels = None
        self.docs = None
        self.shuffle_data()
        self.document_size = len(self.data_set.train_tokens())

    def get_batch_path(self, batch_id) -> str:
        return f'{self.data_path}/data_tensor_{self.data_set_name}_{batch_id}.pt'

    def generate_batches(self):
        if os.path.exists(self.data_path):
            print("Data Set exists")
            return
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.shuffle_data()

        documents = self.token_freqencies(self.docs,
                                          self.data_set.features(),
                                          self.data_set.frequency_map(),
                                          self.alpha)
        while len(documents) < self.batch_size * self.number_of_batches:
            # add empty docs at the end of the last batch for GPU processing
            documents.append([])

        for batch_id in range(self.number_of_batches):
            start_idx = batch_id * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_docs = documents[start_idx:end_idx]
            batch_tensors = torch.stack([self.do_to_sparse_tensor(doc) for doc in batch_docs])
            torch.save(batch_tensors, self.get_batch_path(batch_id))

    def shuffle_data(self):
        docs_copy = self.data_set.train_tokens().copy()
        labels_copy = self.data_set.train_labels().copy()
        paired_docs_labels = list(zip(docs_copy, labels_copy))
        random.seed(self.seed)
        random.shuffle(paired_docs_labels)
        docs, labels = zip(*paired_docs_labels)
        self.docs = docs
        self.labels = labels

    def effective_steps(self):
        return self.number_of_steps // self.squeeze

    def generate_small_batch(self, docs):
        documents = self.token_freqencies(docs,
                                          self.data_set.features(),
                                          {word: 1 for word in self.data_set.features()},
                                          self.alpha)
        return torch.stack([self.do_to_sparse_tensor(doc) for doc in documents])

    def load_batch(self, batch_id) -> torch.sparse:
        batch_path = self.get_batch_path(batch_id)
        return torch.load(batch_path)

    def token_freqencies(self, tokenize_docs: list, features_list: list, frequency_map: dict, alpha: float) -> list:
        documents_dict = {}
        neuron_word_map = {word: idx for idx, word in enumerate(features_list)}
        for doc_nbr, tokens in enumerate(tokenize_docs):
            document_representation = []
            for token in tokens:
                spike_prob = exp(alpha * frequency_map[token])
                document_representation.append((neuron_word_map[token], spike_prob))
            documents_dict[doc_nbr] = document_representation
        return list(documents_dict.values())

    def do_to_sparse_tensor(self, doc):
        spike_idx = 0
        t_idx = 0
        doc_tensor = torch.zeros((self.number_of_steps, len(self.data_set.features()))).to(torch.float16)
        if len(doc) == 0:
            # empty docs to zero tensor
            return self.squeeze_representation(doc_tensor).half().to_sparse()
        while spike_idx < self.number_of_steps:
            if doc[t_idx][1] >= random.random():
                feature_idx = doc[t_idx][0]
                doc_tensor[spike_idx][feature_idx] = 1
                spike_idx += 1
            t_idx += 1
            if t_idx >= len(doc):
                t_idx = 0
        self.squeeze_representation(doc_tensor)
        return self.squeeze_representation(doc_tensor).to_sparse()

    def squeeze_representation(self, doc_tensor):
        doc_tensor = doc_tensor.view(self.number_of_steps // self.squeeze, self.squeeze, len(self.data_set.features()))
        doc_tensor = torch.clamp(doc_tensor.sum(dim=1), 0, 1)
        return doc_tensor
