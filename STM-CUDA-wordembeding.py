import argparse
import csv
import math
import os
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
from evaluation.classification_metrics import ClassificationMetrics
from model.ds.dataset_loader import DatasetLoader
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.gpu.stm_gpu_runner import STMGPUrunner
from model.preprocessing.dataset_interface import DatasetInterface
from model.topic.topic_metric_factory import TopicMetricsFactory
from model.topic.topic_metrics import TopicMetrics
from model.utils.key_value_action import KeyValueAction

from gensim.models.coherencemodel import CoherenceModel


def f_weight(x):
    return x * x * 0.2


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
MODEL_PATH = f'model-output-data/{DATA_SET}-ssntorch-embd'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()

# hyperparameters for data sets

alpha = {'bbc': 0, '20news': -0.0004, 'ag': -0.0004}

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
    Batch_Size = 2225
    neuron_nbr = 60
    loader: CUDADataset = CUDADataset(data_path=MODEL_PATH,
                                      data_set=data_set,
                                      data_set_name=DATA_SET,
                                      number_of_steps=150,
                                      batch_size=Batch_Size,
                                      epochs=1,
                                      alpha=alpha[DATA_SET])
    loader.generate_batches()
    model_name = f'STM_CUDA_{neuron_nbr}'
    training = False
    if training:
        stmModel = STMGPUrunner(FEATURE_LIMIT,
                                neuron_nbr=neuron_nbr,
                                tau_pre=15,
                                tau_post=15,
                                beta=0.8,
                                threshold=0.1,
                                learning_rate=0.001)

        stmModel.train(loader, 20, 50)

        stmModel.save(MODEL_PATH, model_name)

    stmModel = STMGPUrunner.load(MODEL_PATH, model_name)

    topics = stmModel.topics(data_set.features())
    print(topics)
    word2id = Dictionary(data_set.train_tokens())
    cm = CoherenceModel(topics=topics,
                        texts=data_set.train_tokens(),
                        coherence='c_npmi',
                        dictionary=word2id)
    coherence_per_topic = cm.get_coherence_per_topic()
    cm.get_coherence()
    print(cm.get_coherence())
    words = set()
    for t in topics:
        words.update(t)
    print(len(words) / (len(topics) * 10))
    rep = stmModel.encode_docs_dense(loader, 50)
    for r in rep:
        print(r)
    probs_norm = normalize(rep)
    print(probs_norm)
    samples = int(loader.number_of_batches * loader.batch_size)
    train = probs_norm
    test = probs_norm

    clf = LogisticRegression(random_state=0).fit(train, data_set.train_labels())
    pred = clf.predict(test)

    clsf = ClassificationMetrics(pred, data_set.train_labels(), data_set.categories())
    clsf.calculate_results()
    print(clsf.to_fancy_string())

    rep = stmModel.encode_docs_dense_mini_batch(loader, 50, [['star']]).cpu().numpy()
    max_topic = np.argmax(rep)
    print(rep)
    print(np.argmax(rep), topics[max_topic])

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
    #
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