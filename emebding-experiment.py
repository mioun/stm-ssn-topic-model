import argparse
import os

import numpy as np
from bertopic import BERTopic

from ds.dataset_loader import DatasetLoader
from preprocessing.dataset_interface import DatasetInterface
from topic.topic_metric_factory import TopicMetricsFactory
from topic.topic_metrics import TopicMetrics
from utils.key_value_action import KeyValueAction

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
MODEL_PATH = f'model-output-data/{DATA_SET}-embeding'

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()

word_embbeding = np.load(f'{MODEL_PATH}/word_embedding.npy', allow_pickle=True)
similarity_matrix = np.matmul(word_embbeding, word_embbeding.T)

print(len(data_set.features()))
# for line in similarity_matrix:
#     print(line)

n = 10
indices = np.argpartition(similarity_matrix, -n, axis=1)[:, -n:]
print(indices[8159])
words = ['pope']
idxs = [data_set.features().index(w) for w in words]
print(idxs)
for i in range(5000):
    print(indices[i])
    closest = [data_set.features()[s_idx] for s_idx in indices[i][:]]
    print([data_set.features()[i]] + closest
          + [data_set.frequency_map()[data_set.features()[i]]])

doc_embbeding = np.load(f'{MODEL_PATH}/doc_embedding.npy', allow_pickle=True)
topic_model = BERTopic(calculate_probabilities=True, min_topic_size=15)
topics, probs = topic_model.fit_transform([" ".join(doc) for doc in data_set.train_tokens()], doc_embbeding)
print(probs)
print(len(topic_model.get_topic_info()))
N = len(topic_model.get_topic_info())-1
print("M"+str(N))
model_name = f'BERT-TOPIC_{N}_{i}'
# probs_norm = [prob[0:N] for prob in probs]
ENDPOINT_PALMETTO = 'http://palmetto.aksw.org/palmetto-webapp/service/'
metrics: TopicMetrics = TopicMetricsFactory.get_metric('BERT',
                                                       N,
                                                       topic_model,
                                                       ENDPOINT_PALMETTO)

metrics.generate_metrics()
model_name = f'BERT-TOPIC_{N}_{i}'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
topic_model.save(f'{MODEL_PATH}/{model_name}')
metrics.save(MODEL_PATH, f'{model_name}_topic_metrics')

metrics.save(MODEL_PATH, f'{model_name}_topic_metrics')
metrics.save_results_csv(DATA_SET, N, model_name, MODEL_PATH)
