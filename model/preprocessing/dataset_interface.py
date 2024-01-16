from abc import ABC, abstractmethod
from typing import List


class DatasetInterface(ABC):

    @abstractmethod
    def train_tokens(self) -> List:
        """Get tokenized documents"""
        pass

    @abstractmethod
    def test_tokens(self) -> List:
        """Get tokenized documents"""
        pass

    @abstractmethod
    def test_labels(self) -> List:
        """Get test labels"""
        pass

    @abstractmethod
    def train_labels(self) -> List:
        """Get train labels """
        pass

    @abstractmethod
    def features(self) -> List:
        """Get features"""
        pass

    @abstractmethod
    def frequency_map(self) -> dict:
        """Get frequency map"""
        pass

    @abstractmethod
    def categories(self) -> list:
        """ Dataset categories names"""
        pass

    @abstractmethod
    def save(self, folder, name) -> dict:
        """Get frequency map"""
        pass

    @staticmethod
    @abstractmethod
    def load(folder, name):
        """Get frequency map"""
        pass

    @abstractmethod
    def preprocess_data_set(self, docs_training, docs_test, train_labels, test_labels, categories):
        """Preprocess dataset"""
        pass
