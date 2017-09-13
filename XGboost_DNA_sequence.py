import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def convertData(str_list):
    convertL = []
    for ch in str_list:
        if ch == 'D' or ch == 'N' or ch == 'S' or ch == 'R':
            convertL.append(0)
        if ch == 'A':
            convertL.append(1)
        if ch == 'G':
            convertL.append(2)
        if ch == 'T':
            convertL.append(3)
        if ch == 'C':
            convertL.append(4)
    return convertL

def convertY(label_str):
    if label_str == 'EI':
        return 0
    if label_str == 'IE':
        return 1
    if label_str == 'N':
        return 2

crossValidation = False

dataset = pd.read_csv('./splice.csv', header = None)
data = dataset.values
nb_rows = data.shape[0]
seed = 27

X = np.empty([nb_rows, 60])
for idx in range(nb_rows):
    str = data[idx, 2]
    data[idx, 2] = convertData(list(str))
    for c_idx in range(60):
        X[idx, c_idx] = data[idx, 2][c_idx]
    data[idx, 0] = convertY(data[idx, 0])
Y = data[:,0]

test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

dtrain = xgb.DMatrix(X_train, label=y_train)
# param is to be optimized through cross validation
param = {'learning_rate' : 0.1,
 'n_estimators' : 1000,
 'max_depth' : 5,
 'min_child_weight' : 1,
 'gamma' : 0,
 'subsample' : 0.8,
 'colsample_bytree' : 0.8,
 'nthread' : 4,
 'scale_pos_weight' : 1,
 'seed' : 27}
num_round = 10


# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
if crossValidation:
    print ('running cross validation')
    xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed = 0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
if not crossValidation:
    xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
    xgb1.fit(X_train, y_train)
    y_pred1 = xgb1.predict(X_test)
    predictions1 = [round(value) for value in y_pred1]
    nb_test = y_test.shape[0]
    count = 0.0
    for idx in range(nb_test):
        if predictions1[idx] == y_test[idx]:
            count = count + 1
    print('Test accuracy', count/nb_test)
