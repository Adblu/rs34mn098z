import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from utils import remove_nones
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


data = pd.read_excel('dane_oczyszczone.xlsx')

labelencoder = LabelEncoder()

col_list = ['customer_no', 'class', 'channel', 'status', 'emirate', 'code', 'delivery', 'region', 'district',
                'route', '1re1', 'street', 'bldg', 'landmark']

data.columns = col_list

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

sc = StandardScaler()
X = sc.fit_transform(X)


seed = 823
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


print("training")
svclassifier = SVC(kernel='poly')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


print("Accuracy: %.2f%%" % (accuracy_score(y_test, y_pred) * 100))














