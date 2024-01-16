import os.path
import pickle


def save_batch(batch, file_path: str):
    with open(file_path, 'wb') as fp:
        pickle.dump(batch, fp, protocol=4)


def load_batch(file_path):
    with open(file_path, 'rb') as fp:
        batch = pickle.load(fp)
    return batch


def save_data_file(batch, folder_name, file_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    save_batch(batch, os.path.join(folder_name, file_name))


def load_data_file(folder_name, file_name):
    return load_batch(os.path.join(folder_name, file_name))
