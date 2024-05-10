import csv
import os
import time

import numpy as np
import pandas as pd
import pickle as pkl

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

from model.ds.dataset_loader import DatasetLoader
from model.network.stm_model_runner import STMModelRunner
from model.preprocessing.dataset_interface import DatasetInterface
from model.topic.topic_metric_factory import TopicMetricsFactory
from model.topic.topic_metrics import TopicMetrics


def get_model(_model_name):
    if m == 'STM':
        return STMModelRunner.load(OUTPUT_PATH, _model_name)
    if m == 'lda':
        lda_model_file = os.path.join(OUTPUT_PATH, _model_name)
        return LdaModel.load(lda_model_file)
    if m == 'btm':
        btm_model_file = os.path.join(OUTPUT_PATH, _model_name)
        with open(btm_model_file, "rb") as file:
            return pkl.load(file)
def calualte_uniqe(topics):
    uniq = set()
    for t in topics:
        uniq.update(t.words)
    return len(uniq) / (len(topics) * 10)


ENDPOINT_PALMETTO = 'http://localhost:7777/service/'

DATA_SET = '20news'

INPUT_PATH = f'model-input-data/{DATA_SET}'
CONFIG_PATH = 'network-configuration'
OUTPUT_PATH = f'model-output-data/{DATA_SET}-pub'
models = ['STM']
coherence_results_csv = open(os.path.join(OUTPUT_PATH, '_coherence-results.csv'), 'w+')
writer = csv.writer(coherence_results_csv, dialect='unix')
header = ['DS', 'N', 'model', 'TopN', 'metric', 'metric_val']
writer.writerow(header)
N = 20
DATA_SET_PATH = f'model-input-data/{DATA_SET}'
# data_set: DatasetInterface = DatasetLoader(DATA_SET, 5000, DATA_SET_PATH).load_dataset()

for m in models:
    for N in [20, 30, 40]:
        for k in range(5):
            model_name = f'{m}_{N}_{k}'
            print(model_name)
            while not os.path.exists(f'{OUTPUT_PATH}/{model_name}'):
                print("waiting")
                time.sleep(180)
            model = get_model(model_name)
            metrics_file_name = f'{model_name}_topic_metrics'
            if os.path.exists(os.path.join(OUTPUT_PATH, metrics_file_name)):
                metrics: TopicMetrics = TopicMetrics.load(OUTPUT_PATH, f'{model_name}_topic_metrics')
            else:
                metrics: TopicMetrics = TopicMetricsFactory.get_metric(m, N, model, ENDPOINT_PALMETTO)
                metrics.generate_metrics()
                metrics.save(OUTPUT_PATH, f'{model_name}_topic_metrics')
            metrics: TopicMetrics = TopicMetrics.load(OUTPUT_PATH, f'{model_name}_topic_metrics')
            top = sorted(metrics.topics, key=lambda x: x.get_metric('npmi'), reverse=True)
            res_topic = top[:10] + top[-10:]

            # id2word = corpora.Dictionary(data_set.train_tokens())
            # corpus = [id2word.doc2bow(text) for text in data_set.train_tokens()]
            # cm = CoherenceModel(topics=[t.words for t in metrics.topics], texts=data_set.train_tokens(), corpus=corpus,
            #                     dictionary=id2word, coherence='c_npmi')
            # print(f'C_NPMI : {cm.get_coherence()}')
            for top_n in [100, 75]:
                for metric in metrics.coherence_metrics:
                    result_line = [DATA_SET, N, m, top_n, metric,
                                   round(metrics.get_average_metric_for_top_n(metric, top_n), 3)]
                    print(result_line)
                    writer.writerow(result_line)
                    coherence_results_csv.flush()
            for top_n in [100, 75]:
                result_line = [DATA_SET, N, m, top_n, "unique",
                               round(metrics.calualte_uniqe(), 3)]
                print(result_line)
                writer.writerow(result_line)
                coherence_results_csv.flush()
            # tc += metrics.calualte_uniqe() * metrics.get_average_metric_for_top_n('npmi')
            # ni += metrics.get_average_metric_for_top_n('npmi')
            # pu += metrics.calualte_uniqe()
    # print(f'{epoch} npmi {np.round(ni/5,3)} puw : {np.round(pu/5,3)}')
    # print(f'{epoch} tc = {np.round(tc / 5, 3)}')

df_coh = pd.read_csv(os.path.join(OUTPUT_PATH, '_coherence-results.csv'))

print(f'######## {DATA_SET} ######')
npmi_m = {'lda_auto': [], 'STM': [], 'btm': [], 'bert': []}
for model in models:
    mean_all_N = []
    for N in [20, 30, 40]:
        for metric in ['ca', 'npmi', 'unique']:
            m = df_coh[(df_coh['model'] == model) & (df_coh['N'] == N) & (df_coh['metric'] == metric) & (
                    df_coh['TopN'] == 100) & (
                               df_coh['DS'] == DATA_SET)]['metric_val']
            # print(model, N)
            # print(m)
            print(model, N, metric, round(m.mean(), 3), round(m.std(), 3))
            if metric == 'unique':
                npmi_m[model].append(m.mean())
                mean_all_N.append(round(m.mean(), 3))
    print("avg ", model, N, metric, round(np.average(mean_all_N), 3))
print(f'#######################################################', npmi_m)
# for m in models:
#     print(m, np.mean(npmi_m[m]))
