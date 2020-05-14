import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


def adaboost(path, path_test, preds):
    data = pd.read_excel(path)
    test = pd.read_excel(path, path_test)
    data = data.drop(columns=['town-symbol'])
    data = data.drop(columns=['city-size'])
    test = test.drop(columns=['town-symbol'])
    test = test.drop(columns=['city-size'])
    predicts = []
    for i in range(len(preds)):
        y = data[preds[i]]
        X = data.drop(columns=[preds[i]])
        x_test = test.drop(columns=[preds[i]])

        adb = AdaBoostRegressor()
        adb_param_grid = {'n_estimators': [50, 100, 150, 200, 250],  # Number of weak learners to train iteratively.,
                          'learning_rate': [0.001, 0.01, 0.1, 1],
                          # It contributes to the weights of weak learners. It uses 1 as a default value.,
                          'random_state': [1]}

        gsADB = GridSearchCV(adb, param_grid=adb_param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        gsADB.fit(X, y)
        ADB_best = gsADB.best_estimator_
        adb = AdaBoostRegressor(ADB_best)
        adb.fit(X, y)

        predicts.append(adb.predict(x_test))

    results = pd.read_excel(path, 'test-discription')
    len(predicts)
    noremlized_predicts = []

    for i in range(len(preds)):
        noremlized_predicts.append(predicts[i]*results['voted'])
        print(list(noremlized_predicts[i]))

adaboost("predictVotes.xlsx", "test", ["p-votes-MHL","p-votes-PA","p-votes-VDAM"])

