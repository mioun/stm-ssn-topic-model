import argparse
import os

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups

from evaluation.retrival_metrics import RetrivalMetrics
from model.datasets.dataset_loader import DatasetLoader
from model.preprocessing.dataset_interface import DatasetInterface
from model.utils.key_value_action import KeyValueAction
from topic.topic_metric_factory import TopicMetricsFactory
from topic.topic_metrics import TopicMetrics

docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
print(docs[0])
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

DATA_SET_PATH = f'model-input-data/{DATA_SET}'
MODEL_PATH = f'model-output-data/{DATA_SET}-article-bert-full2'

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()
docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
# topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=True)
# topics, probs = topic_model.fit_transform(docs)

ENDPOINT_PALMETTO = 'http://palmetto.aksw.org/palmetto-webapp/service/'

# for txt in data_set.full_dataset():
#     print(txt)


# model = SentenceTransformer("all-mpnet-base-v2")
# embeddings = model.encode(data_set.full_dataset(), show_progress_bar=True)

for N in [20,30,40]:
    for i in range(5):
        # cluster_model = KMeans(n_clusters=N)

        topic_model = BERTopic(language="english",
                               calculate_probabilities=True,
                               min_topic_size=10,
                               verbose=True)
        topics, probs = topic_model.fit_transform([" ".join(doc) for doc in data_set.train_tokens()])
        print(probs)
        print(len(topic_model.get_topic_info()))
        if len(topic_model.get_topic_info()) < N + 1 :
            i = i - 1
            continue
        print(probs)

        model_name = f'BERT-TOPIC_{N}_{i}'
        probs_norm = [prob[0:N] for prob in probs]


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

        metrics.save_results_csv(DATA_SET, N, model_name, MODEL_PATH)

        clustering_met: RetrivalMetrics = RetrivalMetrics(model_name, N, probs_norm, probs_norm,
                                                          data_set.train_labels(), data_set.train_labels(),
                                                          data_set.categories())
        clustering_met.calculate_metrics()

        print("Purity BERT: ", clustering_met.purity)

        clustering_met.save(MODEL_PATH, f'{model_name}')

# for k in range(5):
#
#     for t in topics:
#         print(t)
#     # topic_model.reduce_topics([" ".join(doc) for doc in data_set.train_tokens()], nr_topics=30)
#     # freq = topic_model.get_topic_info()
#     # print("Number of topics: {}".format(len(freq)))
#     # print(freq['Representation'].head())
#     for N in [40, 30, 20]:
#         if not os.path.exists(MODEL_PATH):
#             os.makedirs(MODEL_PATH)
#         topic_model.save(f'{MODEL_PATH}/bert_{N}_{k}')
#
#     # Coherence Evaluation
#
#     metrics.generate_metrics()
#     metrics.save(MODEL_PATH, f'{model_name}_topic_metrics')
#     metrics.save_results_csv(DATA_SET, N, 'BTM', MODEL_PATH)
#
#     train_texts = [' '.join(tokens) for tokens in data_set.train_tokens()]
#     btm_train_vec = btm.get_vectorized_docs(train_texts, model.vocabulary_)
#     train_prob = model.transform(btm_train_vec)
#
#     clustering_met: RetrivalMetrics = RetrivalMetrics(model_name, N, train_prob, train_prob,
#                                                       data_set.train_labels(), data_set.train_labels(),
#                                                       data_set.categories())
#     clustering_met.calculate_metrics()
#
#     print("Purity BTM: ", clustering_met.purity)
#
#     clustering_met.save(MODEL_PATH, f'{model_name}')
#
# freq = topic_model.get_topic_info()
# print("Number of topics: {}".format(len(freq)))
# print(freq['Representation'].head())
# # print(len(topics))
# #
# # if not os.path.exists(MODEL_PATH):
# #     os.makedirs(MODEL_PATH)
# # topic_model.save(f'{MODEL_PATH}/bert')
# for i in range(30):
#     print([x[0] for x in topic_model.get_topic(i)])
# # print(topic_model.get_topic(i))
