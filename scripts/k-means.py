import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data manipulation
X = pd.read_csv('../datasets/x_train_gr_smpl.csv')
y = pd.read_csv('../datasets/y_train_smpl.csv')
from sklearn.utils import shuffle
X = shuffle(X,random_state=3)
y = shuffle(y,random_state=3)

# EDA
print('X: {}'.format(X.shape))
print('y: {}'.format(y.shape))
plt.figure()
y.hist(bins=10)
plt.title('Number of instances of each class')
plt.xlabel('Class')
plt.ylabel('Instances')
plt.show()

# training test split, dev set not req at the moment
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=3)

# normalisation is probably not necessary for kmeans
X_train = X_train / 255
X_test = X_test / 255

# is the size of the data correct?
print('Training Data: {}'.format(X_train.shape))
print('Training labels: {}'.format(y_train.shape))
print('Test Data: {}'.format(X_test.shape))
print('Test labels: {}'.format(y_test.shape))

# check the meaning of life image
plt.figure()
plt.imshow(X_train.values[42].reshape((48,48)))
plt.show()

# Unsupervised kmeans
from sklearn.cluster import KMeans
k = 50 # maybe increase this from 10, although there is 10 classes, signs of one class may look v. different, e.g. at an angle
kmeans = KMeans(n_clusters=k,random_state=3)
# kmeans.fit(X_train)
# y_labels_train = kmeans.labels_
# alternative notation
y_labels_train = kmeans.fit_predict(X_train)


# get the centroids of the clusters
centroids = kmeans.cluster_centers_
images = centroids.reshape(k, 48, 48)
images *= 255

# what does a centroid image look like
plt.figure()
plt.imshow(images[9])
plt.show()

# make relation between (a) y_labels_train and (b) y_train, i.e. label of 1 in (a) corresponds to label of 4 in (b)
# check y_labels_test against y_test for the same relation
def get_corresponding_labels(kmeans, dataset_labels):
    corrected_labels = {}

    for i in range(kmeans.n_clusters):
        labels = []
        index = np.where(kmeans.labels_ == i)
        labels.append(dataset_labels.values[index])

        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        if np.argmax(counts) in corrected_labels:
            corrected_labels[np.argmax(counts)].append(i)
        else:
            corrected_labels[np.argmax(counts)] = [i]
    return corrected_labels

label_dictionary = get_corresponding_labels(kmeans, y_train)

# label correspondonce
label_dictionary

# test the classifier
y_labels_test = kmeans.predict(X_test)  # Classifying them all in the same cluster

# compare y_labels_test to y_test via the label label_dictionary      NOT WORKING ATM

# correct = 0
# wrong = 0
#
# for i in range(len(y_labels_test)):
#     if y_labels_test[i] in label_dictionary[y_test.values.item(i)]:
#         correct += 1
#     else:
#         wrong += 1
#
# print('total num = {} , number checked = {}, percent correct = {}%'.format(len(y_labels_test), correct+wrong, correct / len(y_labels_test)))
