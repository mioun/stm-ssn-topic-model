import os

from model.ds.loaders.base_data_loader import BaseDataLoader

categories = [
    'business', 'entertainment', 'politics', 'sport', 'tech']

categories_mapping = {idx: val for idx, val in enumerate(categories)}
categories_mapping_rew = {val: idx for idx, val in enumerate(categories)}


class BBCLoader(BaseDataLoader):

    def init_dataset(self):
        DATA_FOLDER = './data/bbc'
        for file in os.listdir(DATA_FOLDER):
            class_folder = os.path.join(DATA_FOLDER, file)
            if os.path.isdir(os.path.join(class_folder)):
                texts, labels = [], []
                for file in os.listdir(class_folder):
                    with open(os.path.join(class_folder, file), mode='r') as doc_file:
                        texts.append(doc_file.read().lower())
                        labels.append(categories_mapping_rew[class_folder.split('/')[-1]])
                class_size = len(texts)
                print(f'Class {class_folder} size: {class_size}')
                divider = int(0.6 * class_size)
                self.training_texts.extend(texts[:divider])
                self.training_labels.extend(labels[:divider])
                self.test_texts.extend(texts[divider:])
                self.test_labels.extend(labels[divider:])

    def categories(self):
        return ['business', 'entertainment', 'politics', 'sport', 'tech']




