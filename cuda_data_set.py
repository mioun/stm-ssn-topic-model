import os
import random
from abc import ABC
from math import exp

import torch

from preprocessing.dataset_interface import DatasetInterface


class CUDADataset(ABC):

    def __init__(self, data_path, data_set: DatasetInterface,
                 data_set_name,
                 batch_size,
                 alpha,
                 epochs=1,
                 number_of_steps: int = 150):

        self.data_set_name = data_set_name
        self.number_of_steps = number_of_steps
        self.alpha = alpha
        self.epochs = epochs
        self.data_path = data_path
        self.batch_size = batch_size
        self.data_set = data_set
        self.number_of_batches = len(self.data_set.train_tokens()) // batch_size

    def get_batch_path(self, batch_id) -> str:
        return f'{self.data_path}/data_tensor_{self.data_set_name}_{batch_id}.pt'

    def generate_batches(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        documents = self.token_freqencies(self.data_set.train_tokens(),
                                          self.data_set.features(),
                                          self.data_set.frequency_map(),
                                          self.alpha)
        for batch_id in range(self.number_of_batches):
            start_idx = batch_id * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_docs = documents[start_idx:end_idx]
            batch_tensors = torch.stack([self.do_to_sparse_tensor(doc) for doc in batch_docs])
            torch.save(batch_tensors, self.get_batch_path(batch_id))

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
        sparse_tensors = []
        spike_idx = 0
        t_idx = 0
        doc_tensor = torch.zeros((self.number_of_steps, len(self.data_set.features())))
        while spike_idx < self.number_of_steps:
            if doc[t_idx][1] >= random.random():
                feature_idx = doc[t_idx][0]
                doc_tensor[spike_idx][feature_idx] = 1
                spike_idx += 1
            t_idx += 1
            if t_idx >= len(doc):
                t_idx = 0
        sparse_tensors.append(doc_tensor.to_sparse())
        return doc_tensor.to_sparse()
