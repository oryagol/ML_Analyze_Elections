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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sn



def regression_forest(path, path_test, pred):
    data = pd.read_excel(path)
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

    return predicts


def predict_kosher_votes(path, path_test, pred):
    predicts = regression_forest(path, path_test,pred)

    predicts = 1 - predicts

    results = pd.read_excel(path, 'test-kosher-discription')

    noremlized_predicts = predicts * results['eligble-votes']

    print(list(noremlized_predicts))

    return noremlized_predicts

def predict_disqualified_votes(path, path_test, pred):
    predicts = regression_forest(path, path_test,pred)
    discriptions = pd.read_excel(path, 'test-disqualified-discription')
    boolean_predictions = []
    for i in range(len(predicts)):
        if predicts[i] >= 0.008:
            boolean_predictions.append(True)
        else:
            boolean_predictions.append(False)

    dis = discriptions['disqualified-votes']
    eli_votes = discriptions['eligble-votes']

    true_vals = dis/eli_votes
    boolean_true = []
    for i in range(len(true_vals)):
        if true_vals[i] >= 0.008:
            boolean_true.append(True)
        else:
            boolean_true.append(False)

    cm = confusion_matrix(boolean_true, boolean_predictions)

    ax = plt.subplot()
    sn.heatmap(cm, annot=True, ax = ax, fmt = ".1f")

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Put Observation', 'Dont Put Observation'])
    ax.yaxis.set_ticklabels(['Put Observation', 'Dont Put Observation'])

    plt.show()

    return predicts

predict_disqualified_votes('predictKosherVotes.xlsx', 'test-disqualified', 'p-disqualified-votes')
#predict_kosher_votes('predictKosherVotes.xlsx', 'test-kosher', 'p-disqualified-votes')

