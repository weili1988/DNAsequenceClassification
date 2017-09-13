# DNAsequenceClassification
====================

Overview
--------
Two python code designed to train XGboost and Keras LSTM model for DNA sequence dataset (https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+%28Splice-junction+Gene+Sequence
s%29)

Input in transfered to (nb_samples, 60) where 60 means there are 60 features per sample. Cross validation are used on both XGboost and Keras LSTM model to optimize the architect and parameters. The test data was split before training and is never touched until the final evaluation. 97% test accuracy are achieved by both models

Requirements
------------
* XGboost
* Keras, tensorflow, sk-learn
