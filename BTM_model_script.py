import csv
import os.path
import time

import bitermplus as btm
import numpy as np
import pickle as pkl

from sklearn.preprocessing import normalize

from model.datasets.dataset_loader import DatasetLoader
from model.evaluation.retrival_metrics import RetrivalMetrics
from model.preprocessing.dataset_interface import DatasetInterface
from topic.topic_metric_factory import TopicMetricsFactory
from topic.topic_metrics import TopicMetrics

DATA_SET = 'bbc'
FEATURE_LIMIT = 2000
ITER = 0
TOPIC_NBR = 20

DATA_SET_PATH = f'model-input-data/{DATA_SET}'
MODEL_PATH = f'model-output-data/{DATA_SET}-article-time-comparison'

data_set_name = f'{DATA_SET}_{FEATURE_LIMIT}'
data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()

texts = [' '.join(doc) for doc in data_set.train_tokens()]

# PREPROCESSING
# Obtaining terms frequency in a sparse matrix and corpus vocabulary
X, vocabulary, vocab_dict = btm.get_words_freqs(texts)
tf = np.array(X.sum(axis=0)).ravel()
docs_vec = btm.get_vectorized_docs(texts, vocabulary)
docs_lens = list(map(len, docs_vec))

# Generating biterms
biterms = btm.get_biterms(docs_vec)

# INITIALIZING AND RUNNING MODEL
time_file = open(f'time_report_btm_{DATA_SET}.csv', 'a+')
writer = csv.writer(time_file)

ENDPOINT_PALMETTO = 'http://localhost:7777/service/'
for N in [20]:
    for i in range(5):
        model_name = f'btm_{N}_{i}'
        model = btm.BTM(X, vocabulary, T=N, M=20, alpha=50 / N, beta=0.01)
        start = time.time()
        model.fit(biterms, iterations=150)
        end = time.time()
        print(f'Total time {end - start}')
        writer.writerow([model_name, end - start, DATA_SET])
        time_file.flush()
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        with open(os.path.join(MODEL_PATH, model_name), "wb") as file:
            pkl.dump(model, file)

        with open(os.path.join(MODEL_PATH, model_name), "rb") as file:
            model = pkl.load(file)

        # Coherence Evaluation

        metrics: TopicMetrics = TopicMetricsFactory.get_metric('BTM', N, model, ENDPOINT_PALMETTO)
        metrics.generate_metrics()
        metrics.save(MODEL_PATH, f'{model_name}_topic_metrics')
        metrics.save_results_csv(DATA_SET, N, 'BTM', MODEL_PATH)

        train_texts = [' '.join(tokens) for tokens in data_set.train_tokens()]
        btm_train_vec = btm.get_vectorized_docs(train_texts, model.vocabulary_)
        train_prob = model.transform(btm_train_vec)

        clustering_met: RetrivalMetrics = RetrivalMetrics(model_name, N, train_prob, train_prob,
                                                          data_set.train_labels(), data_set.train_labels(),
                                                          data_set.categories())
        clustering_met.calculate_metrics()

        print("Purity BTM: ", clustering_met.purity)
        print("Fscore BTM: ", clustering_met.classification_metrics.fscore)

        clustering_met.save(MODEL_PATH, f'{model_name}')
