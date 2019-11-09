import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import data_operations
import constants

# load and split data
X = pd.read_csv('../data/x_train_gr_nrm_slc_rnd_smpl.csv')
y = pd.read_csv('../data/y_train_smpl.csv')
y = data_operations.randomize_data(y, constants.SEED)

# unsupervised, don't really need to split?
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3)

# REDUCE DIMENSIONALITY (can use either tSNE or PCA, tSNE takes much longer and need to optimise for perplexity but gives better approximation)
# from sklearn.manifold import TSNE
# X_train_tsne = TSNE(n_components=3, perplexity=30).fit_transform(X_train)
from sklearn.decomposition import PCA
X_train_pca = PCA(n_components=3).fit_transform(X_train)


# UNSUPERVISED KMEANS CLUSTERING
from sklearn.cluster import KMeans
# More than 10 may also be okay, I am currently checking metrics to find the optimal number of clusters
k = 10
kmeans = KMeans(n_clusters=k,random_state=3)
y_labels_train = kmeans.fit_(X_train_pca)

# label the clusters
y_labels_train = kmeans.labels_

# get the centroids of the clusters
centroids = kmeans.cluster_centers_


# CLUSTER EVALUATION

# method 1 gets corresponding classes, 2 & 3 provide some checks
y_train = np.squeeze(y_train)

# method 1 - gets the most common corresponding label from the actual labels
from scipy.stats import mode
labels = np.zeros_like(y_labels_train)
for i in range(k):
    mask = (y_labels_train == i)
    labels[mask] = mode(y_train[mask])[0]

# method 2 - returns a map of label to label and frequency (basically a check on method 1)
label_map = {}
for i in range(k):
    label_counter = {'Class 0':0,'Class 1':0,'Class 2':0,'Class 3':0,'Class 4':0,'Class 5':0,'Class 6':0,'Class 7':0,'Class 8':0,'Class 9':0}
    for idx in range(10128):
        if y_labels_train[idx] == i:
            item = y_train.values[idx]
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
                print('ooopsy')
    label_map[i] = label_counter


# method 3 - ~ give dictionary but not frequency
# def get_corresponding_labels(kmeans, dataset_labels):
#     corrected_labels = {}
#
#     for i in range(kmeans.n_clusters):
#         labels = []
#         index = np.where(kmeans.labels_ == i)
#         labels.append(dataset_labels.values[index])
#
#         if len(labels[0]) == 1:
#             counts = np.bincount(labels[0])
#         else:
#             counts = np.bincount(np.squeeze(labels))
#
#         if np.argmax(counts) in corrected_labels:
#             corrected_labels[np.argmax(counts)].append(i)
#         else:
#             corrected_labels[np.argmax(counts)] = [i]
#
#     return corrected_labels
#
# label_dictionary = get_corresponding_labels(kmeans, y_train)

# check labels
labels
y_labels_train

label_map
