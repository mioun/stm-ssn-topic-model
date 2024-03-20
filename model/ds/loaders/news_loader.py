import os
import random
import string
from typing import List

from model.ds.loaders.base_data_loader import BaseDataLoader

categories = ['alt.atheism',
              'comp.graphics',
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware',
              'comp.windows.x',
              'misc.forsale',
              'rec.autos',
              'rec.motorcycles',
              'rec.sport.baseball',
              'rec.sport.hockey',
              'sci.crypt',
              'sci.electronics',
              'sci.med',
              'sci.space',
              'soc.religion.christian',
              'talk.politics.guns',
              'talk.politics.mideast',
              'talk.politics.misc',
              'talk.religion.misc']

categories_nick_names = {'alt.atheism': 'r-atheism',
                         'comp.graphics': 'com-graphics',
                         'comp.os.ms-windows.misc': 'com-windows',
                         'comp.sys.ibm.pc.hardware': 'com-pc',
                         'comp.sys.mac.hardware': 'com-mac',
                         'comp.windows.x': 'com-windowsX',
                         'misc.forsale': 'forsale',
                         'rec.autos': 'rec-autos',
                         'rec.motorcycles': 'rec-motorcycles',
                         'rec.sport.baseball': 'rec-baseball',
                         'rec.sport.hockey': 'rec-hockey',
                         'sci.crypt': 'sci-crypt',
                         'sci.electronics': 'sci-electronics',
                         'sci.med': 'sci-medicine',
                         'sci.space': 'sci-space',
                         'soc.religion.christian': 'r-christian',
                         'talk.politics.guns': 'pol-gunns',
                         'talk.politics.mideast': 'pol-mideast',
                         'talk.politics.misc': 'pol-politics',
                         'talk.religion.misc': 'r-religion'}

main_categories = [
    'religion',
    'computers',
    'forsale',
    'recreation',
    'sci',
    'politics'
]

categories_mapping = {
    'alt.atheism': 0,
    'comp.graphics': 1,
    'comp.os.ms-windows.misc': 1,
    'comp.sys.ibm.pc.hardware': 1,
    'comp.sys.mac.hardware': 1,
    'comp.windows.x': 1,
    'misc.forsale': 2,
    'rec.autos': 3,
    'rec.motorcycles': 3,
    'rec.sport.baseball': 3,
    'rec.sport.hockey': 3,
    'sci.crypt': 4,
    'sci.electronics': 4,
    'sci.med': 4,
    'sci.space': 4,
    'soc.religion.christian': 0,
    'talk.politics.guns': 5,
    'talk.politics.mideast': 5,
    'talk.politics.misc': 5,
    'talk.religion.misc': 0
}


class NewsDataLoader(BaseDataLoader):

    def categories(self):
        return categories

    def init_dataset(self):
        self.training_texts, self.training_labels = self.news_groups()
        self.test_texts, self.test_labels = self.news_groups(set_type='test')

    def news_groups(self, set_type='train'):
        DATA_FOLDER = './data'
        if set_type == 'train':
            return self.load_data(DATA_FOLDER, categories, 'train', seed=self.seed)
        if set_type == 'test':
            return self.load_data(DATA_FOLDER, categories, 'test', seed=self.seed)

    def load_data(self, path_to_data: str, folders: list, tribe: str, seed=None, lower_case=True) -> [list, list, list]:
        categories_map = {name: idx for idx, name in enumerate(categories, 0)}
        data: list = []
        for folder in folders:
            path_to_folder = os.path.join(path_to_data + '/20news' + tribe, folder)
            data.extend(self.read_folder(path_to_folder, categories_map[folder]))
        if seed is not None:
            print("Shuffle")
            random.Random(seed).shuffle(data)
        if lower_case:
            texts = [val[1].lower() for val in data]
        else:
            texts = [val[1] for val in data]
        cat = [val[0] for val in data]
        return texts, cat

    def read_folder(self, cat_folder_path, category) -> list:
        res = []
        files = os.listdir(cat_folder_path)
        for f in files:
            path_to_file = os.path.join(cat_folder_path, f)
            data_file = open(path_to_file, 'r', encoding='latin1')
            text = data_file.read()
            if len(text) == 0:
                raise Exception("Empty file exception")
            res.append((category, text))
            data_file.close()
        return res

