import os

from model.ds.loaders.ag_loader import AGLoader
from model.ds.loaders.base_data_loader import BaseDataLoader
from model.ds.loaders.bbc_loader import BBCLoader
from model.ds.loaders.news_loader import NewsDataLoader
from model.preprocessing.data_set_clustering import DataSetFullImpl
from model.preprocessing.dataset_interface import DatasetInterface

class DatasetLoader:

    def __init__(self, name: str, features_limit: int, folder: str):
        self.path = folder
        self.features_limit = features_limit
        self.name = name

    def load_dataset(self) -> DatasetInterface:
        data_set_name = f'{self.name}_{self.features_limit}'
        data_set_path = os.path.join(self.path, data_set_name)
        if not os.path.exists(data_set_path):
            print(f'Preprocessing {self.name} dataset limited to {self.features_limit} features')
            loader: BaseDataLoader = self.get_loader(self.name)
            documents_training, labels_training = loader.gat_training_set()
            document_test, labels_test = loader.gat_test_set()
            data_set: DatasetInterface = DataSetFullImpl(self.features_limit)
            data_set.preprocess_data_set(documents_training, document_test, labels_training, labels_test,
                                         loader.categories())
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            data_set.save(self.path, data_set_name)
            return data_set
        else:
            print(f'Loading {self.name} dataset from {self.path}')
            return DataSetFullImpl.load(self.path, data_set_name)

    def get_loader(self, data_set_name: str) -> BaseDataLoader:
        if '20news' == data_set_name:
            return NewsDataLoader()
        if 'bbc' == data_set_name:
            return BBCLoader()
        if 'ag' == data_set_name:
            return AGLoader()

        raise Exception(f'Loader not implemented for {data_set_name}')
