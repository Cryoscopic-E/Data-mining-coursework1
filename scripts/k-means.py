
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_operations
import constants
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# explained variance plot
def pca_explained_variance_plot(data,x1,x2,y1,y2):
    pca = PCA().fit(data)
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    plt.title('German Street Sign Dataset Explained Variance')
    plt.xlim(x1,x2)
    plt.ylim(y1,y2)
    plt.savefig('explained_variance.png')
    plt.show()

# silhouette score plot
def silhouette_score_plot(data, n):
    cluster_silhouette_score = []
    for k in range(2, n+1):
        kmeans = kmeans_clustering(data, k)
        cluster_silhouette_score.append(silhouette_score(data, kmeans.labels_))
    plt.plot(np.arange(2,n+1,1), cluster_silhouette_score)
    plt.xticks(np.arange(2,n+1,1))
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Silhouette score')
    plt.show()

# Cluster Evaluation
def analyzeCluster(clusterDF, labelsDF):
    countByCluster = pd.DataFrame(data=clusterDF['cluster'].value_counts())
    countByCluster.reset_index(inplace=True,drop=False)
    countByCluster.columns = ['cluster','clusterCount']

    preds = pd.concat([labelsDF,clusterDF], axis=1)
    preds.columns = ['trueLabel','cluster']

    countByLabel = pd.DataFrame(data=preds.groupby('trueLabel').count())

    countMostFreq = pd.DataFrame(data=preds.groupby('cluster').agg( \
                        lambda x:x.value_counts().iloc[0]))
    countMostFreq.reset_index(inplace=True,drop=False)
    countMostFreq.columns = ['cluster','countMostFrequent']

    accuracyDF = countMostFreq.merge(countByCluster, \
                        left_on="cluster",right_on="cluster")
    overallAccuracy = accuracyDF.countMostFrequent.sum()/ \
                        accuracyDF.clusterCount.sum()

    accuracyByLabel = accuracyDF.countMostFrequent/ \
                        accuracyDF.clusterCount

    return countByCluster, countByLabel, countMostFreq, \
            accuracyDF, overallAccuracy, accuracyByLabel


# accuracy by number of clusters
def accuracy_by_num_clusters(X, y, n_clusters, n_init, max_iter, tol, random_state, n_jobs):
    kMeans_inertia = \
        pd.DataFrame(data=[],index=range(2,21),columns=['inertia'])
    overallAccuracy_kMeansDF = \
        pd.DataFrame(data=[],index=range(2,21),columns=['overallAccuracy'])

    for n_clusters in range(2,21):
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, \
                    max_iter=max_iter, tol=tol, random_state=random_state, \
                    n_jobs=n_jobs)

        cutoff = 99
        kmeans.fit(X.loc[:,0:cutoff])
        kMeans_inertia.loc[n_clusters] = kmeans.inertia_
        X_train_kmeansClustered = kmeans.predict(X.loc[:,0:cutoff])
        X_train_kmeansClustered = \
            pd.DataFrame(data=X_train_kmeansClustered, index=X_train.index, \
                         columns=['cluster'])

        countByCluster_kMeans, countByLabel_kMeans, countMostFreq_kMeans, \
            accuracyDF_kMeans, overallAccuracy_kMeans, accuracyByLabel_kMeans \
            = analyzeCluster(X_train_kmeansClustered, y)

        overallAccuracy_kMeansDF.loc[n_clusters] = overallAccuracy_kMeans

    return overallAccuracy_kMeansDF


# # accuracy by number of components (PCA)
# def accuracy_by_num_components(X, y, n_clusters,n_init, max_iter, tol, random_state, n_jobs):
#     kMeans_inertia = pd.DataFrame(data=[],index=[9, 49, 99, 199, \
#                         299, 399, 499, 599, 699, 784],columns=['inertia'])
#
#     overallAccuracy_kMeansDF = pd.DataFrame(data=[],index=[9, 49, \
#                         99, 199, 299, 399, 499, 599, 699, 784], \
#                         columns=['overallAccuracy'])
#
#     for cutoffNumber in [9, 49, 99, 199, 299, 399, 499, 599, 699, 784]:
#         kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, \
#                     max_iter=max_iter, tol=tol, random_state=random_state, \
#                     n_jobs=n_jobs)
#
#         cutoff = cutoffNumber
#         kmeans.fit(X.loc[:,0:cutoff])
#         kMeans_inertia.loc[cutoff] = kmeans.inertia_
#         X_kmeansClustered = kmeans.predict(X.loc[:,0:cutoff])
#         X_kmeansClustered = pd.DataFrame(data=X_kmeansClustered, \
#                                     index=X.index, columns=['cluster'])
#
#         countByCluster_kMeans, countByLabel_kMeans, countMostFreq_kMeans, \
#             accuracyDF_kMeans, overallAccuracy_kMeans, accuracyByLabel_kMeans \
#             = analyzeCluster(X_kmeansClustered, y)
#
#         overallAccuracy_kMeansDF.loc[cutoff] = overallAccuracy_kMeans
#
#         return overallAccuracy_kMeansDF.plot()

def get_label_mapping(kmeans, correct_labels):
    label_map = {}
    for i in range(kmeans.n_clusters):
        label_counter = {'Class 0':0,'Class 1':0,'Class 2':0,'Class 3':0,'Class 4':0,'Class 5':0,'Class 6':0,'Class 7':0,'Class 8':0,'Class 9':0}
        for idx in range(len(correct_labels)):
            if kmeans.labels_[idx] == i:
                item = correct_labels.values[idx]
                if item == 0:
                    label_counter['Class 0'] += 1
                elif item == 1:
                    label_counter['Class 1'] += 1
                elif item == 2:
                    label_counter['Class 2'] += 1
                elif item == 3:
                    label_counter['Class 3'] += 1
                elif item == 4:
                    label_counter['Class 4'] += 1
                elif item == 5:
                    label_counter['Class 5'] += 1
                elif item == 6:
                    label_counter['Class 6'] += 1
                elif item == 7:
                    label_counter['Class 7'] += 1
                elif item == 8:
                    label_counter['Class 8'] += 1
                elif item == 9:
                    label_counter['Class 9'] += 1
                else:
                    print('out of range')
        label_map[i] = label_counter
    return label_map

# workbook

X = pd.read_csv('../data/x_train_gr_norm_slc_rnd_smpl.csv')
y = pd.read_csv('../data/y_train_smpl.csv')
y = data_operations.randomize_data(y, constants.SEED)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3)
train_index = range(0,len(X_train))
test_index = range(0,len(X_test))

# pca variance
pca_explained_variance_plot(X_train,0,200,0.5,1 )

# set PCA for PCA at 100 components
n_components = 100
whiten = False
random_state = 42
pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
X_train_PCA = pca.fit_transform(X_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)

# K-means - Accuracy as the number of clusters varies
n_clusters = 20
n_init = 10
max_iter = 300
tol = 0.0001
random_state = 42
n_jobs = 2

accuracy_by_clusters = accuracy_by_num_clusters(X_train_PCA,y_train,n_clusters,n_init,max_iter,tol,random_state,n_jobs)
accuracy_by_clusters.plot().get_figure().savefig('overallAccuracyClusters.png')

# # set PCA for PCA accuracy
# n_components = 784
# whiten = False
# random_state = 42
# pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
# X_train_PCA = pca.fit_transform(X_train)
# X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)
#
# # K-means - Accuracy as the number of components varies
# accuracy_by_components = accuracy_by_num_components(X_train_PCA,y_train,n_clusters,n_init,max_iter,tol,random_state,n_jobs)
# accuracy_by_components.plot().get_figure().savefig('overallAccuracyComponents.png')
#
# # K-means using 100 PCA and X clusters
# n_components = 100
# whiten = False
# random_state = 42
# pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
# X_train_PCA = pca.fit_transform(X_train)
# X_train_PCA = pd.DataFrame(data=X_train_PCA, index=train_index)

n_clusters = 20
n_init = 10
max_iter = 300
tol = 0.0001
random_state = 42
n_jobs = 2
kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, \
            max_iter=max_iter, tol=tol, random_state=random_state, \
            n_jobs=n_jobs)

# fit the model
kmeans.fit_predict(X_train_PCA)
get_label_mapping(kmeans, y_train)
