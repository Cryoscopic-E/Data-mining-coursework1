import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# complete data
#df_pixels = pd.read_csv('../data/x_train_gr_smpl.csv')

#reduced data
#df_pixels = pd.read_csv('../output/reduced_x_train_gr_smpl.csv')
#df_all_classes = pd.read_csv('../data/y_train_smpl.csv')

#randomized reduced
df_pixels = pd.read_csv('../output/shuffled_reduced_x_train_gr_smpl.csv')
df_all_classes = pd.read_csv('../output/shuffled_y_train_smpl.csv')

upper_limit = int(df_pixels.shape[0] * 0.7)

train_sample_pixels = df_pixels[:upper_limit]
test_sample_pixels = df_pixels[upper_limit:]

train_sample_classes = df_all_classes[:upper_limit]
test_sample_classes = df_all_classes[upper_limit:]

gnb = GaussianNB()
gnb.fit(train_sample_pixels, train_sample_classes.values.ravel())
pred = gnb.predict(test_sample_pixels)

with open("../output/shuffled_naive_bayes_reduced.txt","w") as out_text:
    out_text.write(metrics.classification_report(test_sample_classes, pred))
    cm = metrics.confusion_matrix(test_sample_classes, pred)
    out_text.write(np.array2string(cm))
    out_text.write("\n\nAccuracy score: " + str(metrics.accuracy_score(test_sample_classes,pred)))