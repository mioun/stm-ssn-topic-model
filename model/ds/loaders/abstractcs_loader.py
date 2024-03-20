from ds.loaders.base_data_loader import BaseDataLoader


class AbstractsLoader(BaseDataLoader):

    def categories(self):
        return []

    def init_dataset(self):
        self.load_set()

    def load_set(self):
        pass