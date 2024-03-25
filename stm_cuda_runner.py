import argparse
import csv
import math
import os
import pickle
from math import exp
import random
from gensim.corpora.dictionary import Dictionary
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from tqdm import tqdm

import snntorch as snn

import matplotlib.pyplot as plt
import numpy as np

from cuda_data_set import CUDADataset
from ds.loaders.bbc_loader import BBCLoader
from ds.loaders.news_loader import NewsDataLoader
from evaluation.classification_metrics import ClassificationMetrics
from model.ds.dataset_loader import DatasetLoader
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.gpu.stm_gpu_runner import STMGPUrunner
from model.preprocessing.dataset_interface import DatasetInterface
from model.topic.topic_metric_factory import TopicMetricsFactory
from model.topic.topic_metrics import TopicMetrics
from model.utils.key_value_action import KeyValueAction

from gensim.models.coherencemodel import CoherenceModel

from model.preprocessing.dataset_interface_impl import DatasetInterfaceImpl



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
    DATA_SET = '20news'
    FEATURE_LIMIT = 5000

DATA_SET_PATH = f'model-input-data/{DATA_SET}'
CONFIG_PATH = 'network-configuration'
MODEL_PATH = f'model-output-data/{DATA_SET}-ssntorch-final'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()

# hyperparameters for data sets

alpha = {'bbc': -0.003, '20news': -0.0004, 'ag': -0.0004}

tau_scaling = {'bbc': 50, '20news': 75, 'ag': 75}
eta = {'bbc': 0.0006, '20news': 0.0003, 'ag': 0.0003}
learning_window = {'bbc': 50, '20news': 50, 'ag': 50}
epoch = {'bbc': 15, '20news': 8, 'ag': 5}

time_file = open('time_report.csv', 'a+')
writer = csv.writer(time_file)
data = data_set.train_tokens()

ENDPOINT_PALMETTO = 'http://palmetto.aksw.org/palmetto-webapp/service/'

torch.manual_seed(0)
# plt.style.use(['science'])

if __name__ == '__main__':

    DATA_SET_PATH = f'{MODEL_PATH}/data_tensor_{epoch}.pt'
    Batch_Size = 1000
    neuron_nbr = 40
    model_name = f'STM_CUDA_{neuron_nbr}'
    loader: CUDADataset = CUDADataset(data_path=MODEL_PATH,
                                      data_set=data_set,
                                      data_set_name=DATA_SET,
                                      number_of_steps=150,
                                      batch_size=Batch_Size,
                                      epochs=1,
                                      alpha=alpha[DATA_SET])
    loader.generate_batches()

    stmModel = STMGPUrunner(FEATURE_LIMIT,
                            neuron_nbr=neuron_nbr,
                            tau_pre=15,
                            tau_post=15,
                            beta=0.8,
                            threshold=0.1,
                            learning_rate=0.0001)

    stmModel.train(loader, 10, 50)

    stmModel.save(MODEL_PATH, model_name)

    stmModel = STMGPUrunner.load(MODEL_PATH, model_name)

    topics = stmModel.topics(data_set.features(), 10)
    print(topics)


    words = set()
    for t in topics:
        words.update(t)
    print(len(words) / (len(topics) * 10))

    rep = stmModel.encode_docs_dense(loader, 50)
    for r in rep:
        print(r)
    probs_norm = normalize(rep)

    clf = LogisticRegression(random_state=0).fit(probs_norm, loader.labels)
    pred = clf.predict(probs_norm)

    clsf = ClassificationMetrics(loader.labels, pred,
                                 data_set.categories())
    clsf.calculate_results()
    print(clsf.to_fancy_string())

    metrics: TopicMetrics = TopicMetricsFactory.get_metric('STM-CUDA',
                                                           neuron_nbr,
                                                           topics,
                                                           ENDPOINT_PALMETTO)
    metrics.generate_metrics()

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    metrics.save(MODEL_PATH, f'_topic_metrics')

    metrics.save_results_csv(DATA_SET, neuron_nbr, "stm-cuda", MODEL_PATH)


    # number_of_samples = len(probs_norm)
    # phrases = [['play', 'console'], ['skin']]
    # rep = stmModel.encode_docs_dense_mini_batch(loader, 50,
    #                                             phrases).cpu().numpy()
    # norm_rep = normalize(rep)
    # print(norm_rep)
    # print(norm_rep.shape)
    # max_topic = np.argmax(norm_rep)
    # max_topics = np.argsort(rep)[-3:]
    # print(max_topic)
    # print(max_topics.shape)
    # clf = LogisticRegression(random_state=0).fit(probs_norm, data_set.train_labels()[:number_of_samples])
    # pred = clf.predict(probs_norm)
    #
    # clsf = ClassificationMetrics(data_set.train_labels()[:number_of_samples], pred[:number_of_samples],
    #                              data_set.categories())
    # clsf.calculate_results()
    # print(clsf.to_fancy_string())
    #
    # clustering_met: RetrivalMetrics = RetrivalMetrics("cuda-stm", neuron_nbr, probs_norm, probs_norm,
    #                                                   data_set.train_labels(),
    #                                                   data_set.train_labels(),
    #                                                   data_set.categories())
    # clustering_met.calculate_metrics()
    #
    # print("Purity BERT: ", clustering_met.purity)
    # #
    # clustering_met.save(MODEL_PATH, f'cuda-stm')

    # topics_mapping = stmModel.doc_cluster_mapping(loader)
    # for idx in topics_mapping:
    #     print(idx, len(topics_mapping[idx]))
    #
    # for i in range(neuron_nbr):
    #     docs_ids = topics_mapping[i]
    #
    #     # loader_tmp: CUDADataset = CUDADataset(data_path=MODEL_PATH,
    #     #                                       data_set=data_set,
    #     #                                       data_set_name=f'{DATA_SET}_{i}',
    #     #                                       number_of_steps=150,
    #     #                                       batch_size=len(docs_ids),
    #     #                                       epochs=1,
    #     #                                       alpha=-0.003)
    #     # loader_tmp.generate_batches()
    #     stmModel = STMGPUrunner(FEATURE_LIMIT,
    #                             neuron_nbr=10,
    #                             tau_pre=15,
    #                             tau_post=15,
    #                             beta=0.8,
    #                             threshold=0.1,
    #                             learning_rate=0.005)
    #
    #     stmModel.train(loader, 20, 50, docs_ids)
    #     print("#########################")
    #     print(topics[i])
    #     print("|")
    #     print("|")
    #     print("|")
    #     print(stmModel.topics(data_set.features(), 10))
    #     print("#########################")
    words_total = set()
    # for i in range(neuron_nbr):
    #     temporal_ds: DatasetInterface = DatasetInterfaceImpl(feature_limit=5000)
    #     docs_ids = topics_mapping[i]
    #     bbc_loader = NewsDataLoader()
    #
    #     total_docs = bbc_loader.training_texts + bbc_loader.test_texts
    #     lables = [data_set.train_labels()[id] for id in docs_ids]
    #     temporal_docs = [total_docs[id] for id in docs_ids]
    #     temporal_ds.preprocess_data_set(temporal_docs, temporal_docs, lables, lables, categories=data_set.categories())
    #     loader_tmp: CUDADataset = CUDADataset(data_path=MODEL_PATH,
    #                                           data_set=temporal_ds,
    #                                           data_set_name=f'{DATA_SET}_{i}',
    #                                           number_of_steps=150,
    #                                           batch_size=len(temporal_ds.train_tokens()),
    #                                           epochs=1,
    #                                           alpha=-0.01)
    #     loader_tmp.generate_batches()
    #     print(len(temporal_ds.features()))
    #     words_total.update(temporal_ds.features())
    #     words_map = {feature: idx for idx, feature in enumerate(temporal_ds.features())}
    #     print(words_map)
    #     with open(f'{MODEL_PATH}/word_map_{i}.pkl', 'wb') as pickle_file:
    #         pickle.dump(words_map, pickle_file)
    #     with open(f'{MODEL_PATH}/freq_map_{i}.pkl', 'wb') as pickle_file:
    #         pickle.dump(temporal_ds.frequency_map(), pickle_file)
    #     stmModel = STMGPUrunner(len(temporal_ds.features()),
    #                             neuron_nbr=10,
    #                             tau_pre=15,
    #                             tau_post=15,
    #                             beta=0.8,
    #                             threshold=0.1,
    #                             learning_rate=0.001)
    #
    #     stmModel.train(loader_tmp, 30, 50)
    #     stmModel.save(MODEL_PATH, f'partial_net{i}')
    #     print("#########################")
    #     print(topics[i])
    #     print("|")
    #     print("|")
    #     print("|")
    #     print(stmModel.topics(temporal_ds.features(), 10))
    #     print("#########################")
    # for i in range(neuron_nbr):
    #     model = STMGPUrunner.load(MODEL_PATH, f'partial_net{i}')
    #     print(model.net[0].weight.data.shape)

    # print(f'Total words : {len(words_total)}')
    # samples = int(loader.number_of_batches * loader.batch_size)
    # train = probs_norm[:11000]
    # test = probs_norm[11000:samples]
    #
    # clf = LogisticRegression(random_state=0).fit(train, data_set.train_labels()[:11000])
    # pred = clf.predict(test)
    #
    # clsf = ClassificationMetrics(data_set.train_labels()[11000:samples], pred, data_set.categories())
    # clsf.calculate_results()
    # print(clsf.to_fancy_string())
    #
    # clustering_met: RetrivalMetrics = RetrivalMetrics("cuda-stm", neuron_nbr, probs_norm, probs_norm,
    #                                                   data_set.train_labels()[:samples],
    #                                                   data_set.train_labels()[:samples],
    #                                                   data_set.categories())
    # clustering_met.calculate_metrics()
    #
    # print("Purity BERT: ", clustering_met.purity)
    # #
    # clustering_met.save(MODEL_PATH, f'cuda-stm')
    #
    # metrics: TopicMetrics = TopicMetricsFactory.get_metric('STM-CUDA',
    #                                                        neuron_nbr,
    #                                                        topics,
    #                                                        ENDPOINT_PALMETTO)
    # metrics.generate_metrics()
    #
    # if not os.path.exists(MODEL_PATH):
    #     os.makedirs(MODEL_PATH)
    #
    # metrics.save(MODEL_PATH, f'_topic_metrics')
    #
    # metrics.save_results_csv(DATA_SET, neuron_nbr, "stm-cuda", MODEL_PATH)
    # #
    # # #################################Purity#############################################################################
