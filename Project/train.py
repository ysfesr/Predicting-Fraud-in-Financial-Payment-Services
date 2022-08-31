import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow import log_metric, log_param

n_estimators = int(sys.argv[1])
max_samples = float(sys.argv[2])
max_features = float(sys.argv[3])
max_depth = None if sys.argv[4] == 'None' else int(sys.argv[4])

df = pd.read_csv('./dataset/PS_log.csv')

df.drop('Unnamed: 0', axis=1, inplace=True)
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

Y = X['isFraud']
del X['isFraud']

# Eliminate columns shown to be irrelevant for analysis in the EDA
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

# Binary-encoding of labelled data in 'type'
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int) # convert dtype('O') to dtype(int)

X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0), \
      ['oldBalanceDest', 'newBalanceDest']] = - 1

X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0), \
      ['oldBalanceOrig', 'newBalanceOrig']] = -2

X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2)

with mlflow.start_run():
    cls = RandomForestClassifier(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        max_depth=max_depth)
    cls.fit(trainX, trainY)
    y_pred = cls.predict(testX)
    accuracy = accuracy_score(testY, y_pred)
    precision = precision_score(testY, y_pred)
    recall = recall_score(testY, y_pred)
    auc = roc_auc_score(testY, y_pred)

    print('Accuracy score ', accuracy)
    print("Precision score: ", precision)
    print("Recall score: ", recall)
    print("AUC score", auc)

    log_param("n_estimators", n_estimators)
    log_param("max_samples", max_samples)
    log_param("max_features", max_features)
    log_param("max_depth", max_depth)
    
    log_metric("Accuracy", accuracy)
    log_metric("Precision", precision)
    log_metric("Recall", recall)
    log_metric("AUC", auc)
    mlflow.sklearn.log_model(cls, "FraudDetection")
