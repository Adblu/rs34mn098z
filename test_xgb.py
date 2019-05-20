import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils import remove_nones

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

seed = 823
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

gbm = xgb.XGBClassifier(
    predictor='gpu_predictor',
    eta=0.1,
    base_score=0.5,
    n_estimators=100,
    max_leaves=1,
    max_depth=5,
    min_child_weight=10,
    gamma=0.05832865986158353,
    subsample=0.85,
    colsample_bytree=0.8712468101113986,
    colsample_bylevel=0.9,
    reg_lambda=1,
    reg_alpha=1,
    sketch_eps=1,
    objective='reg:logistic',
    nthread=8,
    scale_pos_weight=0.2429,
).fit(X_train, y_train)

y_pred = gbm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
