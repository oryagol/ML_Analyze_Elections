import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor # Import Decision Tree Classifier
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


def regression_forest(path, path_test):
    data = pd.read_excel(path)
    pred = 'p-disqualified-votes'
    test = pd.read_excel(path, path_test)
    y = data[pred]

    data = data.drop(columns=['town-symbol'])
    X = data.drop(columns=[pred])
    test = test.drop(columns=['town-symbol'])
    x_test = test.drop(columns=[pred])

    param_grid = {
        'n_estimators': [200 ,400 ,600, 800, 1000],
        'max_depth': [10, 20, 30, 50, 70],
        'max_features': ['sqrt', X.shape[1]],
        'min_samples_split': [2, 3, 4],
    }

    rfr = RandomForestRegressor(random_state=1)

    best_forest = RandomizedSearchCV(rfr, param_grid, n_jobs=-1, cv=5,
                            n_iter=1, verbose=1, random_state=1)

    best_forest.fit(X, y)

    best_forest.best_params_

    predicts = best_forest.predict(x_test)

    predicts = 1-predicts

    results = pd.read_excel(path, 'test-discription')

    noremlized_predicts = predicts*results['eligble-votes']

    print(list(noremlized_predicts))

    mse = metrics.mean_squared_error(test[pred], predicts)

    print("mse: "+str(mse))

    return mse

regression_forest('predictKosherVotes.xlsx', 'test')