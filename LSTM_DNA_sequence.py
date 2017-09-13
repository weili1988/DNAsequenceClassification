import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Flatten, TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


# build model
def buildModel(nb_lstm, nbTdDense):
    model = Sequential()
    model.add(Conv1D(filters=8, kernel_size=3, input_shape = (60,1), padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(LSTM(nb_lstm, return_sequences = True))
    model.add(Dropout(0.5))
    
    model.add(TimeDistributed(Dense(nbTdDense, activation='relu')))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(3, activation= 'softmax'))
    return model

# data preprocessing, convert 'AGCT' to 1 2 3 4
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
#convert category to 0 1 2
def convertY(label_str):
    if label_str == 'EI':
        return 0
    if label_str == 'IE':
        return 1
    if label_str == 'N':
        return 2


seed = 7 # random seed to repeat result
np.random.seed(seed)
cv = False # if in phase of cross validation ? or real training testing
dataset = pd.read_csv('./splice.csv', header = None) # directory for dataset
nb_epochs = 5 #number of epochs in each training phase

data = dataset.values
nb_rows = data.shape[0]
X = np.empty([nb_rows, 60])

#data preprocessing. X is to be filled
for idx in range(nb_rows):
    str = data[idx, 2]
    data[idx, 2] = convertData(list(str))
    for c_idx in range(60):
        X[idx, c_idx] = data[idx, 2][c_idx]
    data[idx, 0] = convertY(data[idx, 0])
Y = data[:,0]

#prepare X Y, split train and test
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
print(X_train.shape)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
print(X_test.shape)

# cross validation phase:
if cv:
    # split into input (X) and output (Y) variables
    X = X_train
    Y = y_train # cannot use the one- hot version here, later after went into
    
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for [train, test] in kfold.split(X, Y):
        Y_train = np_utils.to_categorical(Y[train])
        Y_test_cv = np_utils.to_categorical(Y[test]) #y test in cross validation
        model = buildModel(128, 64)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        #Fit the model
        model.fit(X[train], Y_train, epochs = nb_epochs, batch_size=16, verbose=0)
        # evaluate the model
        scores = model.evaluate(X[test], Y_test_cv, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))) # the mean and std for 'test' accuracy within cross validation

#### after the CV finished and all parameters and architect is fixed, use all train data to train the weights and do the final evaluation on real untouched test data
if not cv:
    # one hot Y
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)
    model = buildModel(128, 64)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #Fit the model
    model.fit(X_train, Y_train, epochs = nb_epochs, batch_size=16, verbose=0)
    score, acc = model.evaluate(X_test, Y_test, batch_size=16)
    print('Test score:', score)
    print('Test accuracy:', acc)

