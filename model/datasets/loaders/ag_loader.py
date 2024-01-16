import math

import numpy as np
import pandas as pd

from model.datasets.loaders.base_data_loader import BaseDataLoader


class AGLoader(BaseDataLoader):

    def categories(self):
        return ['world', 'sports', 'business', 'sci/tech']

    def init_dataset(self):
        self.load_set()

    def load_set(self):
        texts = []
        labels = []
        df1 = pd.read_csv(f'data/AG/train.csv')
        df2 = pd.read_csv(f'data/AG/test.csv')
        df = pd.DataFrame(np.concatenate([df1.values, df2.values]), columns=df1.columns)
        print(pd.concat([df1, df2]).shape[0])
        df['text'] = df['Title'] + ' ' + df['Description']
        df['Class Index'].replace({4: 0}, inplace=True)
        df.drop(['Title', 'Description'], axis=1, inplace=True)
        for class_id in range(4):
            class_df = df[df['Class Index'] == class_id]
            div = math.floor(class_df.shape[0] * 0.6)
            total_texts = [x[1].lower() for x in class_df.values.tolist()]
            total_labels = [x[0] for x in class_df.values.tolist()]
            self.training_texts.extend(total_texts[:div])
            self.training_labels.extend(total_labels[:div])
            self.test_texts.extend(total_texts[div:])
            self.test_labels.extend(total_labels[div:])
        list_date = df.values.tolist()[1:]
        for row in list_date:
            texts.append(row[1].lower())
            labels.append(row[0])
        return texts, labels


loader = AGLoader()
test_t, test_l = loader.gat_test_set()
train_t, train_l = loader.gat_training_set()
print(len(train_t))
print(len(test_t))
print(len(test_t) + len(train_t))
print(len(train_l))
print(len(test_l))
print(len(test_l) + len(train_l))
print(test_t[0])
