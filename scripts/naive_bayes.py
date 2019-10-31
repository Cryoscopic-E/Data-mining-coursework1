import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# complete data
#df_pixels = pd.read_csv('../data/x_train_gr_smpl.csv')

#reduced data
#df_pixels = pd.read_csv('../output/reduced_x_train_gr_smpl.csv')
#df_all_classes = pd.read_csv('../data/y_train_smpl.csv')

#randomized complete 
df_pixels = pd.read_csv('../output/shuffled_x_train_gr_smpl.csv')

#randomized reduced
#df_pixels = pd.read_csv('../output/shuffled_reduced_x_train_gr_smpl.csv')

#feature selected
df_pixels_feats = pd.read_csv('../output/smpl_2_features.csv')
df_all_classes = pd.read_csv('../output/shuffled_y_train_smpl.csv')

upper_limit = int(df_pixels.shape[0] * 0.7)

#train_sample_pixels = df_pixels[:upper_limit]
test_sample_pixels = df_pixels[upper_limit:]

train_sample_classes = df_all_classes[:upper_limit]
test_sample_classes = df_all_classes[upper_limit:]

gnb = GaussianNB()
gnb.fit(df_pixels_feats, df_all_classes.values.ravel())
pred = gnb.predict(df_pixels)

with open("../output/feature_naive_bayes_complete.txt","w") as out_text:
    out_text.write(metrics.classification_report(df_all_classes, pred))
    cm = metrics.confusion_matrix(df_all_classes, pred)
    out_text.write(np.array2string(cm))
    out_text.write("\n\nAccuracy score: " + str(metrics.accuracy_score(df_all_classes,pred)))