import argparse
import csv
import os
import time

import numpy as np

from model.datasets.dataset_loader import DatasetLoader
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.network.stm_model_runner import STMModelRunner
from model.preprocessing.dataset_interface import DatasetInterface
from model.topic.topic_metric_factory import TopicMetricsFactory
from model.topic.topic_metrics import TopicMetrics

from model.utils.key_value_action import KeyValueAction

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
MODEL_PATH = f'model-output-data/{DATA_SET}-pub'

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()

# hyperparameters for data sets

alpha = {'bbc': -0.003, '20news': -0.0003, 'ag': -0.0004}

tau_scaling = {'bbc': 50, '20news': 75, 'ag': 75}
eta = {'bbc': 0.0006, '20news': 0.0003, 'ag': 0.0003}
learning_window = {'bbc': 50, '20news': 50, 'ag': 50}
epoch = {'bbc': 15, '20news': 8, 'ag': 5}

time_file = open('time_report.csv', 'a+')
writer = csv.writer(time_file)
data = data_set.train_tokens()

doc_lenght = []
for doc in data:
    doc_lenght.append(len(doc))
print(len(data))
print(np.median(doc_lenght))

ENDPOINT_PALMETTO = 'http://localhost:7777/service/'

for N in [20, 30, 40]:
    for i in range(5):
        model_name = f'STM_{N}_{i}'

        model = STMModelRunner(feature_limit=FEATURE_LIMIT,
                               feature_list=data_set.features(),
                               freq_map=data_set.frequency_map(),
                               total_texts=len(data_set.train_tokens()),
                               encoder_size=N,
                               inh=5,
                               config_path=CONFIG_PATH,
                               alpha=alpha[DATA_SET],
                               learning_window=learning_window[DATA_SET],
                               tau_scaling=tau_scaling[DATA_SET],
                               minimum_spikes=150,
                               eta=eta[DATA_SET])
        train_tmp = []

        for i in range(15):
            train_tmp.extend(data_set.train_tokens())
        start_time = time.time()

        model.train(1, train_tmp)

        total_time = time.time() - start_time
        writer.writerow([model_name, total_time, DATA_SET])
        time_file.flush()
        model.save(MODEL_PATH, model_name)

        # Coherence Evaluation

        metrics: TopicMetrics = TopicMetricsFactory.get_metric('STM', N, model, ENDPOINT_PALMETTO)
        metrics.generate_metrics()
        metrics.save(MODEL_PATH, f'{model_name}_topic_metrics')
        metrics.save_results_csv(DATA_SET, N, 'STM', MODEL_PATH)

        # Purity Evaluation

        model = STMModelRunner.load(MODEL_PATH, model_name)
        train_represent = model.represent_norm(data_set.train_tokens())
        clustering_met: RetrivalMetrics = RetrivalMetrics(model_name, N, train_represent, train_represent,
                                                          data_set.train_labels(), data_set.train_labels(),
                                                          data_set.categories())
        clustering_met.calculate_metrics()
        print("Purity STM: ", clustering_met.purity)
        clustering_met.save(MODEL_PATH, model_name)
