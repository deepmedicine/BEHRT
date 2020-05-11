import os
import _pickle as pickle
import h5py


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class H5Recorder(object):
    def __init__(self, path):
        self.path = path

    def open(self, read=False):
        if read:
            self.hf = h5py.File(self.path, 'r')
        else:
            self.hf = h5py.File(self.path, 'w')

    def write(self, key, value):
        self.hf.create_dataset(key, data=value)

    def read(self, key):
        return self.hf.get(key)

    def close(self):
        self.hf.close()