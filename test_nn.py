import keras.optimizers
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import one_hot_encode_object_array, get_prepared_data
from sklearn.externals import joblib
import pickle

X, Y = get_prepared_data('dane_oczyszczone.xlsx')

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

joblib.dump(scaler, "scaler.save")
Y = one_hot_encode_object_array(Y)

seed = 128
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
X_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size / 2, random_state=seed)

classifier = Sequential()
mid = 'elu'
end_act = 'sigmoid'
classifier.add(Dense(8, activation=mid, input_dim=X.shape[1]))
classifier.add(Dense(16, activation=mid, kernel_initializer='random_uniform'))
classifier.add(Dense(64, activation=mid, kernel_initializer='random_uniform'))
classifier.add(Dense(128, activation=mid, kernel_initializer='random_uniform'))
classifier.add(Dense(64, activation=mid, kernel_initializer='random_uniform'))
classifier.add(Dense(32, activation=mid, kernel_initializer='random_uniform'))
classifier.add(Dense(Y.shape[1], activation=end_act, kernel_initializer='random_uniform'))

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0, decay=0.0, amsgrad=False)

classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)

score, accuracy = classifier.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100))

Xt = scaler.transform(x_val)

scoret, accuracyt = classifier.evaluate(Xt, y_val)
print("Accuracy: %.2f%%" % (accuracyt * 100))

# pickle.dump(classifier, open('classifier_model.pkl','wb'))
classifier.save('classifier_model.h5')