import argparse
import csv
import math
import os
from math import exp
import random

import numpy as np
import torch
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from tqdm import tqdm

from cuda_data_set import CUDADataset
from model.ds.dataset_loader import DatasetLoader
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.gpu.stm_gpu_runner import STMGPUrunner
from model.preprocessing.dataset_interface import DatasetInterface
from model.utils.key_value_action import KeyValueAction
from model.topic.topic_metric_factory import TopicMetricsFactory
from model.topic.topic_metrics import TopicMetrics

dtype = torch.float
device = torch.device("cuda")


def to_token_freq_dict(tokenize_docs: list, features_list: list, frequency_map: dict, alpha: float) -> dict:
    documents_dict = {}
    neuron_word_map = {word: idx for idx, word in enumerate(features_list)}
    for doc_nbr, tokens in enumerate(tokenize_docs):
        document_representation = []
        for token in tokens:
            spike_prob = exp(alpha * frequency_map[token])
            document_representation.append((neuron_word_map[token], spike_prob))
        documents_dict[doc_nbr] = document_representation
    return documents_dict


def spike_tensor(documents_probability_dic: dict, spike_limit):
    sparse_tensors = []
    docs = list(documents_probability_dic.values())
    for idx, doc in tqdm(enumerate(docs)):
        spike_idx = 0
        t_idx = 0
        doc_tensor = torch.zeros((spike_limit, FEATURE_LIMIT))
        while spike_idx < spike_limit:

            if doc[t_idx][1] >= random.random():
                feature_idx = doc[t_idx][0]
                doc_tensor[spike_idx][feature_idx] = 1
                spike_idx += 1
            t_idx += 1
            if t_idx >= len(doc):
                t_idx = 0
        sparse_tensors.append(doc_tensor.to_sparse())
    return torch.stack(sparse_tensors)


#######################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--params',
                    nargs='*',
                    action=KeyValueAction)
params = parser.parse_args().params

if params:
    print('Parameters from request')
    DATA_SET = params['data_set']
    FEATURE_LIMIT = int(params['features_limit'])
else:
    DATA_SET = 'bbc'
    FEATURE_LIMIT = 2000

DATA_SET_PATH = f'model-input-data/{DATA_SET}'
CONFIG_PATH = 'network-configuration'
MODEL_PATH = f'model-output-data/{DATA_SET}-final-results-pre'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()

# hyperparameters for data sets

alpha = {'bbc': -0.003, '20news': -0.0004, 'ag': -0.0004}

tau_scaling = {'bbc': 50, '20news': 75, 'ag': 75}
eta = {'bbc': 0.0006, '20news': 0.0001, 'ag': 0.0003}
learning_window = {'bbc': 50, '20news': 50, 'ag': 50}
epoch = {'bbc': 15, '20news': 20, 'ag': 5}

time_file = open('time_report.csv', 'a+')
# writer = csv.writer(time_file)
# data = data_set.train_tokens()

ENDPOINT_PALMETTO = 'http://localhost:7777/service/'

# plt.style.use(['science'])
epochs = {"ag": 7, "bbc": 20, "20news": 20}
etas = {"ag": 0.0005, "bbc": 0.001, "20news": 0.0001}
batch_size = {"ag": 2000, "bbc": 2225, "20news": 2000}

for word in data_set.features():
    print(word, data_set.frequency_map()[word])

squeezing_results = open(f'{MODEL_PATH}/final_results.csv', "a+")
csv_writer = csv.writer(squeezing_results)
if __name__ == '__main__':
    print(f'{DATA_SET} batch size: {batch_size[DATA_SET]} eta: {etas[DATA_SET]} epochs: {epochs[DATA_SET]}')
    for i in range(5):
        for N in [20]:
            results_line = [N]
            Batch_Size = batch_size[DATA_SET]
            neuron_nbr = N
            squueze = 3
            model_name = f'STM_CUDA_{neuron_nbr}_{i}'
            DATA_SET_PATH = f'{MODEL_PATH}/{Batch_Size}_{squueze}'

            loader: CUDADataset = CUDADataset(data_path=DATA_SET_PATH,
                                              data_set=data_set,
                                              data_set_name=DATA_SET,
                                              number_of_steps=150,
                                              batch_size=Batch_Size,
                                              squeeze=squueze,
                                              epochs=1,
                                              alpha=alpha[DATA_SET]
                                              )
            loader.generate_batches()

            stmModel = STMGPUrunner(FEATURE_LIMIT,
                                    neuron_nbr=neuron_nbr,
                                    tau_pre=15,
                                    tau_post=15,
                                    beta=0.8,
                                    threshold=0.5,
                                    learning_rate=0.0001)

            stmModel.train(loader, 20)

            stmModel.save(MODEL_PATH, model_name)
            stmModel = STMGPUrunner.load(MODEL_PATH, model_name)
            #
            topics = stmModel.topics(data_set.features(), 10)
            print(topics)

            rep = stmModel.encode_docs(loader)
            clustering_met: RetrivalMetrics = RetrivalMetrics(model_name,
                                                              neuron_nbr,
                                                              rep,
                                                              rep,
                                                              loader.labels,
                                                              loader.labels,
                                                              data_set.categories())
            clustering_met.calculate_metrics()

            print("Purity STM: ", clustering_met.purity)
            print("SI: ", clustering_met.calculate_silhouette_score())
            clustering_met.save(MODEL_PATH, f'cuda-stm')
            results_line.append(np.round(clustering_met.purity, 3))
            metrics: TopicMetrics = TopicMetricsFactory.get_metric('STM-CUDA', N, topics, ENDPOINT_PALMETTO)
            metrics.generate_metrics()

            results_line.append(np.round(metrics.calualte_uniqe(), 3))
            results_line.append(np.round(metrics.get_average_metric_for_top_n('npmi'), 3))
            results_line.append(np.round(metrics.get_average_metric_for_top_n('ca'), 3))
            results_line.append(np.round(metrics.get_average_metric_for_top_n('cv'), 3))

            csv_writer.writerow(results_line)
            squeezing_results.flush()

            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)

            metrics.save(MODEL_PATH, f'{model_name}_topic_metrics')
            metrics.save_results_csv(DATA_SET, neuron_nbr, model_name, MODEL_PATH)

            word2id = Dictionary(data_set.train_tokens())

            cm = CoherenceModel(topics=topics,
                                texts=data_set.train_tokens(),
                                coherence='c_npmi',
                                dictionary=word2id)

            coherence_per_topic = cm.get_coherence_per_topic()
            print(coherence_per_topic)
            cm.get_coherence()
            print(cm.get_coherence())

            words = set()
            for t in topics:
                words.update(t)
            puw = len(words) / (len(topics) * 10)
            print(f'PUW : {puw}')
