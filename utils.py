import numpy as np
from keras.utils import np_utils


def remove_nones(data):
    for entry in data.columns:
        data = data[data[entry] != 'none']
    return data


def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))
