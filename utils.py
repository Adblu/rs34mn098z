import numpy as np
from keras.utils import np_utils
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import types
import tempfile
import keras.models
from flask import make_response, request, current_app
from functools import update_wrapper
from datetime import timedelta

cols_list = ['class', 'channel', 'emirate', 'code', 'delivery', 'region', 'district',
             'route', '1re1', 'street', 'bldg', 'landmark']

def remove_nones(data):
    for entry in data.columns:
        data = data[data[entry] != 'none']
    return data


def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))


def get_prepared_data(path):
    data = pd.read_excel(path)

    labelencoder = LabelEncoder()

    data.columns = ['customer_no', 'class', 'channel', 'status', 'emirate', 'code', 'delivery', 'region', 'district',
                    'route', '1re1', 'street', 'bldg', 'landmark']

    data = remove_nones(data)

    data['channel'] = labelencoder.fit_transform(data['channel'].values)
    data['class'] = labelencoder.fit_transform(data['class'].values)
    data['emirate'] = labelencoder.fit_transform(data['emirate'].values)
    data['code'] = labelencoder.fit_transform(data['code'].values)
    data['delivery'] = labelencoder.fit_transform(data['delivery'].values)
    data['region'] = labelencoder.fit_transform(data['region'].values)
    data['district'] = labelencoder.fit_transform(data['district'].values)
    data['route'] = labelencoder.fit_transform(data['route'].astype(str))
    data['1re1'] = labelencoder.fit_transform(data['1re1'].astype(str))
    data['street'] = labelencoder.fit_transform(data['street'].astype(str))
    data['bldg'] = labelencoder.fit_transform(data['bldg'].astype(str))
    data['landmark'] = labelencoder.fit_transform(data['landmark'].astype(str))
    data = data.drop(columns=['customer_no'])

    Y = data.loc[:, data.columns == 'status']
    X = data.loc[:, data.columns != 'status']

    return X, Y

def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator