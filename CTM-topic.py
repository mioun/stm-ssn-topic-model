import argparse
import os

import nltk
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from nltk.corpus import stopwords as stop_words
nltk.download('stopwords')
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from sklearn.datasets import fetch_20newsgroups

from ds.dataset_loader import DatasetLoader
from ds.loaders.bbc_loader import BBCLoader
from preprocessing.dataset_interface import DatasetInterface
from topic.topic_metric_factory import TopicMetricsFactory
from topic.topic_metrics import TopicMetrics
from utils.key_value_action import KeyValueAction

qt = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")

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
    DATA_SET = '20news'
    FEATURE_LIMIT = 5000

DATA_SET_PATH = f'model-input-data/{DATA_SET}'
MODEL_PATH = f'model-output-data/{DATA_SET}-article-ctm'

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()
docs = [" ".join(doc) for doc in data_set.train_tokens()]
# topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=True)
# topics, probs = topic_model.fit_transform(docs)
data_loader = BBCLoader()
documents = data_loader.training_texts+data_loader.test_texts


stopwords = list(stop_words.words("english"))

sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=stopwords)
preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()


ENDPOINT_PALMETTO = 'http://palmetto.aksw.org/palmetto-webapp/service/'

training_dataset = qt.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=768, n_components=20,num_epochs=300)  # 50 topics

ctm.fit(training_dataset)  # run the model

N = 20
metrics: TopicMetrics = TopicMetricsFactory.get_metric('CTM', N, ctm,
                                                       ENDPOINT_PALMETTO)

metrics.generate_metrics()
model_name = f'CTM_{N}_{0}'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
ctm.save(f'{MODEL_PATH}/bert')
metrics.save(MODEL_PATH, f'{model_name}_topic_metrics')

metrics.save_results_csv(DATA_SET, N, model_name, MODEL_PATH)

print(ctm.get_topics())
for idx in ctm.get_topics():
    print(ctm.get_topics()[idx])
