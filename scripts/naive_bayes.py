import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# complete data
#df_pixels = pd.read_csv('../data/x_train_gr_smpl.csv')

#reduced data
df_pixels = pd.read_csv('../output/reduced_x_train_gr_smpl.csv')
df_all_classes = pd.read_csv('../data/y_train_smpl.csv')

gnb = GaussianNB()
gnb.fit(df_pixels, df_all_classes.values.ravel())
pred = gnb.predict(df_pixels)

with open("../output/naive_bayes_reduced.txt","w") as out_text:
    out_text.write(metrics.classification_report(df_all_classes, pred))
    cm = metrics.confusion_matrix(df_all_classes, pred)
    out_text.write(np.array2string(cm))
    out_text.write("\n\nAccuracy score: " + str(metrics.accuracy_score(df_all_classes,pred)))