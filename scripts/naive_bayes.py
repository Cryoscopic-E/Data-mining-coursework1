import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


df_pixels = pd.read_csv('../data/x_train_gr_smpl.csv')
df_all_classes = pd.read_csv('../data/y_train_smpl.csv')

gnb = GaussianNB()
gnb.fit(df_pixels, df_all_classes.values.ravel())
pred = gnb.predict(df_pixels)

print(metrics.classification_report(df_all_classes, pred))
print(metrics.confusion_matrix(df_all_classes, pred))
