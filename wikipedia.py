import argparse
import os

import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from ctfidf import CTFIDFVectorizer
from ds.dataset_loader import DatasetLoader
from preprocessing.dataset_interface import DatasetInterface
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
    DATA_SET = '20news'
    FEATURE_LIMIT = 5000

DATA_SET_PATH = f'model-input-data/{DATA_SET}'
CONFIG_PATH = 'network-configuration'
MODEL_PATH = f'model-output-data/{DATA_SET}-ssntorch'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

data_set: DatasetInterface = DatasetLoader(DATA_SET, FEATURE_LIMIT, DATA_SET_PATH).load_dataset()

train = [" ".join(doc) for doc in data_set.train_tokens()[:11000]]
labels = data_set.train_labels()[:11000]

# Get data
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes', 'stopwords'))
docs = pd.DataFrame({'Document': train, 'Class': labels})
docs_per_class = docs.groupby(['Class'], as_index=False).agg({'Document': ' '.join})

# Create bag of words
count_vectorizer: CountVectorizer = CountVectorizer().fit(docs_per_class.Document)
count = count_vectorizer.transform(docs_per_class.Document)
words = count_vectorizer.get_feature_names_out()

# Extract top 10 words
ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(docs)).toarray()
words_per_class = {newsgroups.target_names[label]: [words[index] for index in ctfidf[label].argsort()[-100:]] for label
                   in docs_per_class.Class}

for i in words_per_class:
    print(i,words_per_class[i])
