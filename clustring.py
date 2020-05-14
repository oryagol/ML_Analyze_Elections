import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import seaborn as sns
from xlrd import open_workbook
from xlutils.copy import copy
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
sns.set()

def GMMAnalyze(path):
    df = pd.read_excel (path)
    df = df.drop(columns=['town-symbol'])
    gmm = GaussianMixture(n_components=6).fit(df)
    scores = gmm.predict(df)
    probs = gmm.predict_proba(df)
    results_df = pd.read_excel ('townsSymbols.xlsx')
    results_df['Cluster'] = scores
    results_df.to_excel('clusteringScoresGMM.xlsx')

def kMeansAnalyze(path):
    df = pd.read_excel (path)
    df = df.drop(columns=['town-symbol'])
    kmeans = KMeans(6, random_state=0)
    scores = kmeans.fit(df).predict(df)
    centroids = kmeans.cluster_centers_
    results_df = pd.read_excel ('townsSymbols.xlsx')
    results_df['Cluster'] = scores
    results_df.to_excel('clusteringScoresKmeans.xlsx')


def hierarchyAnalyze(path):
    df = pd.read_excel(path)
    df = df.drop(columns=['town-symbol'])
    hc = AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='complete')
    scores = hc.fit_predict(df)
    results_df = pd.read_excel('townsSymbols.xlsx')
    results_df['Cluster'] = scores
    results_df.to_excel('clusteringScoresHierarchy.xlsx')


def elbow_met(df):
    from scipy.spatial.distance import cdist
    # k means determine k
    distortions = []
    K = range(1,10)
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



hierarchyAnalyze('ElectionPatternParties.xlsx')


