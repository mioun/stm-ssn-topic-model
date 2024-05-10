from abc import ABC, abstractmethod
from typing import List


class BaseDataLoader(ABC):

    def __init__(self, seed=666, division=None):
        self.seed = seed
        self.division = division
        self.training_texts = []
        self.training_labels = []
        self.test_texts = []
        self.test_labels = []
        self.init_dataset()

    def gat_training_set(self) -> (List[str], List):
        return self.training_texts, self.training_labels

    def gat_test_set(self) -> (List[str], List):
        return self.test_texts, self.test_labels

    def get_stats(self):
        print(f'train size : {len(self.training_texts)}')
        print(f'test size : {len(self.test_texts)}')
        print(f'Unique categories : {len(set(self.test_labels))}')
        print()

    @abstractmethod
    def init_dataset(self):
        pass

    @abstractmethod
    def categories(self):
        pass
