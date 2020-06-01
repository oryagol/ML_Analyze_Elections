import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import seaborn as sns
from xlrd import open_workbook
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
sns.set()
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import DecisionTreeRegressor # Import Decision Tree Classifier
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sn
from sklearn.cluster import MeanShift
from sklearn import preprocessing


# section 2 of assignment, correlation matrix and scatter matrix of the 3 elections
def describeData():
    data_21 = pd.read_csv('resultsKneset21.csv')
    data_22 = pd.read_csv('resultsKneset22.csv')
    data_23 = pd.read_csv('resultsKneset23.csv')

    data_21 = data_21.drop(columns=['town-symbol'])
    data_21 = data_21.drop(columns=['proper-votes'])
    data_21 = data_21.drop(columns=['available-votes'])
    data_21 = data_21.drop(columns=['OTHER'])
    corr = data_21.corr()
    plt.figure(figsize=(10, 10))

    sns.heatmap(corr, vmax=.8, linewidths=0.01,
                square=True, annot=True, cmap='YlGnBu', linecolor="white")
    plt.title('Correlation between features');

    plt.show()

    data_22 = data_22.drop(columns=['town-symbol'])
    data_22 = data_22.drop(columns=['proper-votes'])
    data_22 = data_22.drop(columns=['available-votes'])
    data_22 = data_22.drop(columns=['OTHER'])
    corr = data_22.corr()
    plt.figure(figsize=(10, 10))

    sns.heatmap(corr, vmax=.8, linewidths=0.01,
                square=True, annot=True, cmap='YlGnBu', linecolor="white")
    plt.title('Correlation between features');

    plt.show()

    data_23 = data_23.drop(columns=['town-symbol'])
    data_23 = data_23.drop(columns=['proper-votes'])
    data_23 = data_23.drop(columns=['available-votes'])
    data_23 = data_23.drop(columns=['OTHER'])
    corr = data_23.corr()
    plt.figure(figsize=(10, 10))

    sns.heatmap(corr, vmax=.8, linewidths=0.01,
                square=True, annot=True, cmap='YlGnBu', linecolor="white")
    plt.title('Correlation between features');

    plt.show()

    pd.plotting.scatter_matrix(data_21, figsize=(30, 30),
                               marker='o', hist_kwds={'bins': 20}, s=60,
                               alpha=.8)
    plt.show()

    pd.plotting.scatter_matrix(data_22, figsize=(30, 30),
                               marker='o', hist_kwds={'bins': 20}, s=60,
                               alpha=.8)
    plt.show()

    pd.plotting.scatter_matrix(data_23, figsize=(30, 30),
                               marker='o', hist_kwds={'bins': 20}, s=60,
                               alpha=.8)
    plt.show()

# section 3, GMM algorithm
def GMMAnalyze(path):
    df = pd.read_excel(path)
    df = df.drop(columns=['town-symbol'])
    gmm = GaussianMixture(n_components=6).fit(df)
    scores = gmm.predict(df)
    probs = gmm.predict_proba(df)
    results_df = pd.read_excel('townsSymbols.xlsx')
    results_df['Cluster'] = scores
    results_df.to_excel('clusteringScoresGMM.xlsx')
    print("Finished GMM")

# section 3, K means algorithm
def kMeansAnalyze(path):
    df = pd.read_excel(path)
    df = df.drop(columns=['town-symbol'])
    kmeans = KMeans(6, random_state=0)
    scores = kmeans.fit(df).predict(df)
    centroids = kmeans.cluster_centers_
    print(centroids)
    results_df = pd.read_excel('townsSymbols.xlsx')
    results_df['Cluster'] = scores
    results_df.to_excel('clusteringScoresKmeans.xlsx')
    print("Finished K means")

# section 3, Hierarchical Agglomerative algorithm
def hierarchyAnalyze(path):
    df = pd.read_excel(path)
    df = df.drop(columns=['town-symbol'])
    hc = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='complete')
    scores = hc.fit_predict(df)
    results_df = pd.read_excel('townsSymbols.xlsx')
    results_df['Cluster'] = scores
    results_df.to_excel('clusteringScoresHierarchy.xlsx')
    print("Finished Hierarchical Agglomerative")

# section 3, mean shift algorithm
def meanShiftAnalyze(path):
    df = pd.read_excel(path)
    df = df.drop(columns=['town-symbol'])
    ms = MeanShift()
    ms.fit(df)
    cluster_centers = ms.cluster_centers_
    print(cluster_centers)

    clustering = MeanShift(bandwidth=0.34).fit(df)

    plt.show()

    results_df = pd.read_excel('townsSymbols.xlsx')
    results_df['Cluster'] = clustering.labels_
    results_df.to_excel('clusteringScoresMeanshift.xlsx')

    print("Finished Mean Shift")

# section 3, elbow function for K means
def elbow_met(df):
    from scipy.spatial.distance import cdist
    # k means determine k
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(df)
        kmeanModel.fit(df)
        distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Within groups sum of squares')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

# regression forest for section 4 and 6
def regression_forest(path, path_test, pred):
    data = pd.read_excel(path)
    test = pd.read_excel(path, path_test)
    y = data[pred]

    data = data.drop(columns=['town-symbol'])
    X = data.drop(columns=[pred])
    test = test.drop(columns=['town-symbol'])
    x_test = test.drop(columns=[pred])
    features = list(X.columns)

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

    model =  RandomForestRegressor(n_estimators=best_forest.best_params_['n_estimators'],
                               random_state=1,
                               max_features = best_forest.best_params_['max_features'],
                                   max_depth=best_forest.best_params_['max_depth'],
                                min_samples_split = best_forest.best_params_['min_samples_split'],
                               n_jobs=-1, verbose = 1)

    model.fit(X, y)

    predicts = model.predict(x_test)

    fi_model = pd.DataFrame({'feature': features,
                             'importance': model.feature_importances_}). \
        sort_values('importance', ascending=False)
    fi_model.head(10)

    print(fi_model)

    return predicts

# predict valid votes for section 4
def predict_kosher_votes(path, path_test, pred):
    predicts = regression_forest(path, path_test, pred)

    predicts = 1 - predicts

    results = pd.read_excel(path, 'test-kosher-discription')

    noremlized_predicts = predicts * results['eligble-votes']

    mse = metrics.mean_squared_error(noremlized_predicts, results['kosher-votes'])
    print("mse is: "+str(mse))

    print(['haifa', 'ayelet ha shachar', 'sakhnin', 'eilat','katzerin'])
    print(list(noremlized_predicts))

    return noremlized_predicts

# predict disqualified votes for section 6 with confusion matrix
def predict_disqualified_votes(path, path_test, pred):
    predicts = regression_forest(path, path_test,pred)
    discriptions = pd.read_excel(path, 'test-disqualified-discription')
    test = pd.read_excel(path, 'test-disqualified')
    boolean_predictions = []
    for i in range(len(predicts)):
        if predicts[i]*test['city-size'][i] >= 20 and predicts[i] >= 0.008:
            boolean_predictions.append(True)
        else:
            boolean_predictions.append(False)

    dis = discriptions['disqualified-votes']
    eli_votes = discriptions['eligble-votes']

    true_vals = dis/eli_votes
    boolean_true = []
    for i in range(len(true_vals)):
        if true_vals[i]*eli_votes[i] >= 20 and true_vals[i] >= 0.008:
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
# predicting votes for parties in section 5
def adaboost(path, path_test, preds, parties):
    data = pd.read_excel(path)
    test = pd.read_excel(path, path_test)
    data = data.drop(columns=['town-symbol'])
    test = test.drop(columns=['town-symbol'])
    data = data.drop(columns=['city-size'])
    test = test.drop(columns=['city-size'])
    data = data.drop(columns=['p-votes-SHS'])
    test = test.drop(columns=['p-votes-SHS'])
    data = data.drop(columns=['p-votes-G'])
    test = test.drop(columns=['p-votes-G'])
    data = data.drop(columns=['p-votes-AMT'])
    test = test.drop(columns=['p-votes-AMT'])
    data = data.drop(columns=['p-votes-TB'])
    test = test.drop(columns=['p-votes-TB'])
    data = data.drop(columns=['p-votes-L'])
    test = test.drop(columns=['p-votes-L'])
    data = data.drop(columns=['p-total-voted'])
    test = test.drop(columns=['p-total-voted'])

    predicts = []
    results = pd.read_excel(path, 'test-discription')
    j = 3
    for i in range(len(preds)):
        y = data[preds[i]]
        X = data
        x_test = test
        features = list(X.columns)

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

        fi_model = pd.DataFrame({'feature': features,
                                 'importance': adb.feature_importances_}). \
            sort_values('importance', ascending=False)
        fi_model.head(10)

        print("feature importance of "+str(preds[i]))
        print(fi_model)

        predicts.append(adb.predict(x_test))

        mse = metrics.mean_squared_error(results[parties[i]], predicts[i])
        print("mse is: " + str(mse))
        j += 1

    noremlized_predicts = []

    for i in range(len(preds)):
        noremlized_predicts.append(predicts[i]*results['voted'])
        print("predictions of "+str(preds[i]))
        print(['jerusalem', 'bnei brack', 'sakhnin', 'karmiel', 'daliet el karmel'])
        print(list(noremlized_predicts[i]))



# section 2
#describeData()
# section 3
#kMeansAnalyze("ElectionPatternParties.xlsx")
# section 3
#meanShiftAnalyze("ElectionPatternParties.xlsx")
# section 4
predict_kosher_votes("predictKosherVotes.xlsx", 'test-kosher', 'p-disqualified-votes')
# section 5
#adaboost("predictVotes.xlsx", 'test', ['p-votes-MHL', 'p-votes-PA', 'p-votes-VDAM'],  ['MHL', 'PA', 'VDAM'])
# section 6
#predict_disqualified_votes("predictKosherVotes.xlsx", 'test-disqualified', 'p-disqualified-votes')



