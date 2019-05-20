import pandas as pd
import keras.optimizers
import numpy as np
from keras.utils import np_utils
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))


data = pd.read_excel('dane_oczyszczone.xlsx')

labelencoder = LabelEncoder()

data.columns = ['customer_no', 'class', 'channel', 'status', 'emirate', 'code', 'delivery', 'region', 'district',
                'route', '1re1', 'street', 'bldg', 'landmark']

data['customer_no'] = labelencoder.fit_transform(data['customer_no'].values)
# data['customer_no'] = data['customer_no'].astype(str)
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

Y = data.loc[:, data.columns == 'status']
X = data.loc[:, data.columns != 'status']

sc = StandardScaler()
X = sc.fit_transform(X)
Y = one_hot_encode_object_array(Y)

seed = 823
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

classifier = Sequential()

classifier.add(Dense(4, activation='sigmoid', kernel_initializer='random_uniform', input_dim=X.shape[1]))
classifier.add(Dense(16, activation='sigmoid', kernel_initializer='random_uniform'))
classifier.add(Dense(32, activation='sigmoid', kernel_initializer='random_uniform'))
classifier.add(Dense(Y.shape[1], activation='softmax', kernel_initializer='random_uniform'))

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0, decay=0.0, amsgrad=False)
classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=64, epochs=100, verbose=1)

score, accuracy = classifier.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100))
