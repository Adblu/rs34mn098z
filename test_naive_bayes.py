import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from utils import remove_nones, one_hot_encode_object_array
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_excel('dane_oczyszczone.xlsx')

labelencoder = LabelEncoder()

data.columns = ['customer_no', 'class', 'channel', 'status', 'emirate', 'code', 'delivery', 'region', 'district',
                'route', '1re1', 'street', 'bldg', 'landmark']

data = remove_nones(data)

data['customer_no'] = labelencoder.fit_transform(data['customer_no'].values)
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

seed = 30
test_size = 0.5
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

# predict(iris.data)
predicted = gnb.predict(y_test)

print("Accuracy: %.2f%%" % (accuracy_score(y_test, predicted) * 100))
