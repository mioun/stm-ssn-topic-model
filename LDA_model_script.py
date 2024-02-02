import argparse
import csv
import os
import time

import numpy as np
from gensim import corpora, models
from gensim.models import LdaModel
from sklearn.preprocessing import normalize

from model.datasets.dataset_loader import DatasetLoader
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.preprocessing.dataset_interface import DatasetInterface
from model.utils.key_value_action import KeyValueAction
from model.topic.topic_metric_factory import TopicMetricsFactory
from model.topic.topic_metrics import TopicMetrics

parser = argparse.ArgumentParser()
parser.add_argument('--params',
                    nargs='*',
                    action=KeyValueAction)
params = parser.parse_args().params

if params:
    print(params)
    DATA_SET = params['data_set']
    FEATURE_LIMIT = int(params['features_limit'])
    topic_nbr = int(params['t'])

else:
    DATA_SET = 'bbc'
    FEATURE_LIMIT = 2000

EPOCHS = 30
DATA_SET_PATH = f'model-input-data/{DATA_SET}'
MODEL_PATH = f'model-output-data/{DATA_SET}'

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()
print(len(data_set.train_labels()))
print(len(data_set.train_tokens()))
id2word = corpora.Dictionary(data_set.train_tokens())
corpus = [id2word.doc2bow(text) for text in data_set.train_tokens()]
alpha = 'auto'

time_file = open(f'time_report_lda_{DATA_SET}.csv', 'a+')
writer = csv.writer(time_file)

# it is recommended to use local Palemtto service please follow the instruction on
# https://github.com/dice-group/Palmetto/wiki/How-Palmetto-can-be-used
ENDPOINT_PALMETTO = 'http://palmetto.aksw.org/palmetto-webapp/service/'

for N in [20,30,40]:
    for i in range(8):
        model_name = f'lda_{N}_{i}'
        print(f'Running : {i} for topic conf : {N} for alpha: {alpha} ')
        start = time.time()
        lda_model: LdaModel = models.ldamodel.LdaModel(corpus, id2word=id2word, num_topics=N,
                                                       chunksize=len(data_set.train_tokens()),
                                                       alpha=alpha, eval_every=None, passes=150
                                                       )
        end = time.time()
        print(f'Total time {end - start}')
        writer.writerow([model_name, end - start, DATA_SET])
        time_file.flush()

        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)

        lda_model.save(os.path.join(MODEL_PATH, model_name))

        # Coherence Evaluation

        metrics: TopicMetrics = TopicMetricsFactory.get_metric('LDA', N, lda_model, ENDPOINT_PALMETTO)
        metrics.generate_metrics()
        metrics.save(MODEL_PATH, f'{model_name}_topic_metrics')
        metrics.save_results_csv(DATA_SET, N, 'LDA', MODEL_PATH)

        # Purity Evaluation

        train_dat = [lda_model[id2word.doc2bow(doc)] for doc in data_set.train_tokens()]
        train_porb = np.zeros((len(train_dat), N))
        for idx, vec in enumerate(train_dat):
            for pair in vec:
                train_porb[idx][pair[0]] = pair[1]
        train_norm = normalize(train_porb)

        clustering_met: RetrivalMetrics = RetrivalMetrics(model_name, N, train_norm, train_norm,
                                                          data_set.train_labels(), data_set.train_labels(),
                                                          data_set.categories())
        clustering_met.calculate_metrics()

        print("Purity LDA: ", clustering_met.purity)
        print("Fscore LDA: ", clustering_met.classification_metrics.fscore)

        clustering_met.save(MODEL_PATH, f'{model_name}')

print(f'Finished iter :  {iter} for topic number : {N}')
