import numpy as np
import pandas as pd
import csv
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from sklearn import metrics


"""

Data is loaded in, combined into one large matrix, and randomised. Data takes this form:

Class 0? | Class 1? | Class 2? | ... | Class 9? | Class Number |        Pixel Data           |
    1    |     1    |     0    | ... |    1     |      2       | [231, 294, 183 , ... , 222] |


Note: If you want to work in weka, i can export it to a csv file. Then just convert to .arff format
Note: Replace filepath with your own one to the data

"""

df_pixels = pd.read_csv('../data/x_train_gr_smpl.csv')
df_all_classes = pd.read_csv('../data/y_train_smpl.csv')
df_is_class0 = pd.read_csv('../data/y_train_smpl_0.csv')
# df_is_class1 = pd.read_csv('.data/y_train_smpl_1.csv')
# df_is_class2 = pd.read_csv('/home/msc/odm1/Documents/ML_CW1/y_train_smpl_2.csv')
# df_is_class3 = pd.read_csv('/home/msc/odm1/Documents/ML_CW1/y_train_smpl_3.csv')
# df_is_class4 = pd.read_csv('/home/msc/odm1/Documents/ML_CW1/y_train_smpl_4.csv')
# df_is_class5 = pd.read_csv('/home/msc/odm1/Documents/ML_CW1/y_train_smpl_5.csv')
# df_is_class6 = pd.read_csv('/home/msc/odm1/Documents/ML_CW1/y_train_smpl_6.csv')
# df_is_class7 = pd.read_csv('/home/msc/odm1/Documents/ML_CW1/y_train_smpl_7.csv')
# df_is_class8 = pd.read_csv('/home/msc/odm1/Documents/ML_CW1/y_train_smpl_8.csv')
# df_is_class9 = pd.read_csv('/home/msc/odm1/Documents/ML_CW1/y_train_smpl_9.csv')
#
#
# combined_data = np.column_stack((df_is_class0, df_is_class1, df_is_class2, df_is_class3,
#                                  df_is_class4, df_is_class5, df_is_class6, df_is_class7,
#                                  df_is_class8, df_is_class9, df_all_classes, df_pixels))

#
# np.random.shuffle(combined_data)

# class Row:
#     def __init__(self, array):
#         self.pixels = np.array(array[11:],dtype=float) # 2304 pixels
#         self.target = float(array[10]) #0 to 9
#         self.labels = np.array(array[:10], dtype=bytes)
#
#
# def save(number):
#     with open('csv/random_rows_'+str(number)+'.csv', newline='', mode='w') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in combined_data:
#             writer.writerow(i)
#
#
#
# def load(csvFile):
#     temp = []
#     with open('csv/' + csvFile,mode='r') as csvFile:
#         reader = csv.reader(csvFile)
#         for i in reader:
#             temp.append(Row(i))
#     pass
#
# rows = load('random_rows_0.csv')
# print(rows[0].target)
#
# class0, class1, class2, class3, class4, class5, class6, class7, class8, class9, all_class, pixels = \
#     np.hsplit(combined_data, range(1,12) )
#
#
# all_class.transpose()

# print(pixels.shape)


"""

Display an image!

"""

# pd_to_np_image = combined_data[3576][11:]
# image = np.reshape(pd_to_np_image, (48, 48))
# print(combined_data[3576][:11])
#
# cv2.imwrite("imagetest.jpg", image);


"""

<<Naive Bayes>>

Here, we are selecting roughly ~75% of the dataset and using it to train the model. The remaining 25% is used
to test the model.


"""

# gnb = GaussianNB()
# gnb.fit(pixels[:8000],all_class[:8000])
#
# prediction_all = gnb.predict(pixels[8000:])
#
#
# print(metrics.classification_report(all_class[8000:], prediction_all))
# print(metrics.confusion_matrix(all_class[8000:], prediction_all))



"""

<<Feature Selection>>

"""

# =============================================================================
# print(df_pixels.iloc[0])
# 
# print('correlating ... ')
# cor = df_pixels[0:100].corr()
# #print(cor.to_numpy()[:9,:9])
# 
# cor_target = abs(cor)
# 
# relevate_features= cor_target[cor_target<1]
# print(relevate_features.sort_values(ascending=True)[2294:])
# =============================================================================


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


transformer = SelectKBest(score_func=chi2, k=10)

#new_data = transformer.fit_transform(df_pixels.values, df_is_class0)
fit = transformer.fit(df_pixels, df_is_class0)
scores = pd.DataFrame(fit.scores_)
columns = pd.DataFrame(df_pixels.columns)
#concat 2 dataframes for better visualization
featuresScore = pd.concat([columns,scores],axis=1)
featuresScore.columns = ['Pixel','Score']

print(featuresScore.nlargest(10,'Score'))