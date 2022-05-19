import getopt, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
import mlflow
from mlflow import log_metric, log_param

argumentList = sys.argv[1:]
options = "k:c:"
long_options = ["kernel=", "C="]

arguments, values = getopt.getopt(argumentList, options, long_options)

for currentArgument, currentValue in arguments:
      if currentArgument in ('-k',"--kernel"):
            kernel = str(currentValue)
      elif currentArgument in ('-c', '--C'):
            c_value = int(currentValue)
try:
      print(f"Kernel: {kernel}, C: {c_value}")
except:
      print('You must specify the kernel and C arguments')


df = pd.read_csv('./dataset/PS_log.csv')
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

mlflow.set_tracking_uri("http://localhost:5000")
mlflow_experiment_id = 0

with mlflow.start_run(experiment_id=mlflow_experiment_id):
    svc = SVC(kernel=kernel, C=c_value)
    svc.fit(trainX, trainY)
    y_pred = svc.predict(testX)
    accuracy = accuracy_score(testY, y_pred)
    precision = precision_score(testY, y_pred)
    recall = recall_score(testY, y_pred)
    auc = roc_auc_score(testY, y_pred)

    print('Accuracy score ', accuracy)
    print("Precision score: ", precision)
    print("Recall score: ", recall)
    print("AUC score", auc)

    log_param('kernel', kernel)
    log_param('C', c_value)
    log_metric("Accuracy", accuracy)
    log_metric("Precision", precision)
    log_metric("Recall", recall)
    log_metric("AUC", auc)
    mlflow.sklearn.log_model(svc, "FraudDetection")
