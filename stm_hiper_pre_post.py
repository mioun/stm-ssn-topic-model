import argparse
import csv
import os
import time

import numpy as np
from brian2 import set_device, device
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from model.ds.dataset_loader import DatasetLoader
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.network.stm_model_runner import STMModelRunner
from model.preprocessing.dataset_interface import DatasetInterface
from model.topic.topic_metric_factory import TopicMetricsFactory
from model.topic.topic_metrics import TopicMetrics

from model.utils.key_value_action import KeyValueAction
from sklearn.metrics import silhouette_score

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
MODEL_PATH = f'model-output-data/{DATA_SET}-spike-pre-post'

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()

# hyperparameters for data sets

alpha = {'bbc': -0.003, '20news': -0.0003, 'ag': -0.0004}

tau_scaling = {'bbc': 50, '20news': 75, 'ag': 75}
eta = {'bbc': 0.0006, '20news': 0.0003, 'ag': 0.0003}
learning_window = {'bbc': 50, '20news': 50, 'ag': 50}
epoch = {'bbc': 15, '20news': 8, 'ag': 5}

data = data_set.train_tokens()

doc_lenght = []
for doc in data:
    doc_lenght.append(len(doc))
print(len(data))
print(np.median(doc_lenght))

# it is recommended to use local Palemtto service please follow the instruction on
# https://github.com/dice-group/Palmetto/wiki/How-Palmetto-can-be-used
ENDPOINT_PALMETTO = 'http://palmetto.aksw.org/palmetto-webapp/service/'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
time_file = open(f'{MODEL_PATH}/tau_prepost_results.csv', 'a+')
writer = csv.writer(time_file)

for N in [30]:
    for i in range(3):
        for tau_pre in [1, 5, 10, 20, 30, 50, 100]:
            for tau_post in [1, 5, 10, 20, 30, 50, 100]:
                model_name = f'STM_{N}_{tau_pre}_{tau_post}_{i}'
                row = [N, tau_pre, tau_post, i, ]
                model = STMModelRunner(feature_limit=FEATURE_LIMIT,
                                       feature_list=data_set.features(),
                                       freq_map=data_set.frequency_map(),
                                       total_texts=len(data_set.train_tokens()),
                                       encoder_size=N,
                                       inh=5,
                                       config_path=CONFIG_PATH,
                                       alpha=alpha[DATA_SET],
                                       learning_window=tau_pre,
                                       tau_scaling=tau_post,
                                       minimum_spikes=150,
                                       eta=eta[DATA_SET],
                                       compilation_path=f'{MODEL_PATH}/brian_output')
                train_tmp = []

                # training is faster in batches in such case, training can be done by adding several times data set to the batch
                # here the data set is added 15 times to batch == 15 training epoch

                for _ in range(epoch[DATA_SET]):
                    train_tmp.extend(data_set.train_tokens())
                start_time = time.time()

                TRAINING_EPOCHS = 1
                model.train(TRAINING_EPOCHS, train_tmp)

                model.save(MODEL_PATH, model_name)

                model: STMModelRunner = STMModelRunner.load(MODEL_PATH, model_name)
                # Coherence Evaluation

                word2id = Dictionary(data_set.train_tokens())
                topics = [t.words for t in model.extract_topics_from_model(10)]
                cm = CoherenceModel(topics=topics,
                                    texts=data_set.train_tokens(),
                                    coherence='c_npmi',
                                    dictionary=word2id)
                coherence_per_topic = cm.get_coherence_per_topic()
                print(coherence_per_topic)
                cm.get_coherence()
                row.append(np.round(cm.get_coherence(), 3))

                words = set()
                for t in topics:
                    words.update(t)
                puw = len(words) / (len(topics) * 10)

                row.append(np.round(puw, 3))

                # metrics.save(MODEL_PATH, f'{model_name}_topic_metrics')
                # metrics.save_results_csv(DATA_SET, N, 'STM', MODEL_PATH)
                #
                # met: TopicMetrics = TopicMetrics.load(MODEL_PATH, f'{model_name}_topic_metrics')

                # cm = CoherenceModel(topics=topics,
                #                     texts=data_set.train_tokens(),
                #                     coherence='c_npmi',
                #                     dictionary=word2id)
                # coherence_per_topic = cm.get_coherence_per_topic()
                # cm.get_coherence()
                # print(cm.get_coherence())

                # for t in met.topics:
                #     print(f'{t.metrics["npmi"]}, {t.metrics["ca"]}, {t.words}')

                # Purity Evaluation
                with open(f'{MODEL_PATH}/{model_name}_rep.npy', 'wb') as f:
                    train_represent = model.represent_norm(data_set.train_tokens())
                    np.save(f, train_represent)
                with open(f'{MODEL_PATH}/{model_name}_rep_dense.npy', 'wb') as f:
                    model = STMModelRunner.load(MODEL_PATH, model_name)
                    train_represent = model.represent_norm(data_set.train_tokens(), 1)
                    np.save(f, train_represent)
                with open(f'{MODEL_PATH}/{model_name}_rep.npy', 'rb') as f:
                    train_represent = np.load(f)

                clustering_met: RetrivalMetrics = RetrivalMetrics(model_name, N, train_represent, train_represent,
                                                                  data_set.train_labels(), data_set.train_labels(),
                                                                  data_set.categories())
                clustering_met.calculate_metrics()
                print(clustering_met.calculate_silhouette_score(data_set.train_tokens()))
                print("Purity STM: ", clustering_met.purity)
                clustering_met.save(MODEL_PATH, model_name)
                row.append(np.round(clustering_met.purity, 4))
                row.append(np.round(clustering_met.calculate_silhouette_score(data_set.train_tokens()), 4))
                writer.writerow(row)
                time_file.flush()
