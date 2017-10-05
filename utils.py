import os
import time
import struct
import fnmatch
import numpy as np
from math import log, floor

def human_format(number):
    """
    From https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings-in-python
    """
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '{}{}'.format(int(number / k**magnitude), units[magnitude])

def name_file(config):
    a = '3d'
    if config.num_dimensions == 2:
        a = '2d'
    b = human_format(config.num_samples)
    return '/embeddings_{}_{}'.format(a, b)

def prepare_dirs(config):
    for path in [config.data_dir, config.plot_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def str2bool(v):
    return v.lower() in ('true', '1')

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def load_data(path):

    pattern = '*ubyte'

    # crawl directory and grab filenames
    names = []
    for path, subdirs, files in os.walk(path):
        for filename in files:
            if fnmatch.fnmatch(filename, pattern):
                names.append(os.path.join(path, filename))
                
    num_files = len(names)
        
    # read the files into a numpy array
    data = {}
    for i in range(num_files):
        if 'train' in names[i]:
            if 'images' in names[i]:
                data['train_imgs'] = read_idx(names[i]) 
            else:
                data['train_labels'] = read_idx(names[i])
        else:
            if 'images' in names[i]:
                data['test_imgs'] = read_idx(names[i])
            else:
                data['test_labels'] = read_idx(names[i])

    X_train = data['train_imgs']
    X_test = data['test_imgs']
    y_train = data['train_labels']
    y_test = data['test_labels']

    X_train = np.expand_dims(X_train, axis=1).astype('float32')
    X_test = np.expand_dims(X_test, axis=1).astype('float32')

    return X_train, y_train, X_test, y_test
