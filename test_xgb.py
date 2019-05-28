import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils import remove_nones, get_prepared_data
import pickle

X, Y = get_prepared_data('dane_oczyszczone.xlsx')

seed = 823
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=seed)

gbm = xgb.XGBClassifier(
    predictor='gpu_predictor',
    eta=0.1,
    base_score=0.5,
    n_estimators=100,
    max_leaves=1,
    max_depth=4,
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
).fit(X_train.values, y_train.values)

y_pred = gbm.predict(X_test.values)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%% on test set." % (accuracy * 100.0))

accuracy = accuracy_score(y_val, gbm.predict(X_val.values))
print("Accuracy: %.2f%% on validation set." % (accuracy * 100.0))

gbm._Booster.save_model('xgb_model.bst')
